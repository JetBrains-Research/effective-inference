from transformers import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import prune_conv1d_layer, find_pruneable_heads_and_indices

def hidden_to_heads(x, config):
    num_attention_heads = config.attention_config.num_heads
    attention_head_size =  config.attention_config.d_model // config.attention_config.num_heads
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x

class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        self.config = config
        self.features = config['features']
        self.device = config['device']
        self.batch_size = config['batch_size']

        self.dim_size = config['d_model']
        if config.split_heads:
            self.dim_size = config['d_model'] // config['num_heads']
                        
        for k in self.features:                
            if 'hidden' in k:
                learnable_parameters = f'torch.nn.Linear(in_features={self.dim_size}, out_features=1, bias=False)'
                exec(f"self.{k} = {learnable_parameters}")
            else:
                learnable_parameters = f'nn.Parameter(torch.randn(1), requires_grad=True)'
                exec(f"self.{k} = {learnable_parameters}")
                    
            
    def forward(self, seq_len_arg=None, **kwargs):
        if seq_len_arg is not None:
            result = torch.zeros((self.batch_size, seq_len_arg, seq_len_arg), device=self.device)
            # print(self.batch_size)
            
            
            for arg_name, arg_value in kwargs.items():
                namespace = {'cur_result': None, 'self': self, 'arg_name': arg_name, 'arg_value': arg_value}
                if 'hidden' in arg_name:
                    exec(f"cur_result = self.{arg_name}(arg_value)", namespace)
                else:
                    exec(f"cur_result = self.{arg_name} * arg_value", namespace)
                if 'from' in arg_name:
                    result += torch.mean(namespace['cur_result'], dim=0, keepdim=True).T
                else:
                    # print(*result.shape)
                    # print(f"cur_result = self.{arg_name}(arg_value)")
                    # print(arg_value.shape)
                    # print(namespace['cur_result'].shape)
                    result += torch.mean(namespace['cur_result'], dim=0, keepdim=True)
        else:
            for arg_name, arg_value in kwargs.items():
                bs = len(arg_value)
                break
            result = torch.zeros((bs, 1), device=self.device, )
            
            for arg_name, arg_value in kwargs.items():
                namespace = {'cur_result': None, 'self': self, 'arg_name': arg_name, 'arg_value': arg_value}
                if 'hidden' in arg_name:
                    exec(f"cur_result = self.{arg_name}(arg_value)", namespace)
                else:
                    exec(f"cur_result = self.{arg_name} * arg_value", namespace)
                
                result += namespace['cur_result'].view((bs, 1))
            
        return result


from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model

class LinearClassifierGPTAttention(GPT2Attention):
    """
    Idea: attention weights are predicted by Linear Classifier
    """
    def __init__(self, gpt2_config, config):
        super(LinearClassifierGPTAttention, self).__init__(gpt2_config)
        self.config = config
        self.linear_config = config.attention_config

        if self.linear_config.split_heads or self.linear_config.model_for_each_head:
            for head_num in range(self.linear_config['num_heads']):
                learnable_parameters = f'LinearAttention(self.config.attention_config)'
                namespace = {'head_num': head_num, 'self': self, 'LinearAttention': LinearAttention}
                exec(f"self.linear_model_{head_num} = {learnable_parameters}", namespace)
        else:
            self.linear_model = LinearAttention(config.attention_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            # special_tokens_idxs: Optional[List[int]] = [0]
    ) -> Tuple[torch.Tensor]:
        
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if self.config.attention_config.split_heads:
            hidden_states = hidden_to_heads(hidden_states, self.config)

            attentions = []
            for head_num in range(self.linear_config['num_heads']):
                seq_len = hidden_states.shape[1]
                positions = torch.arange(seq_len).view(-1, 1)
                
                full_data_to_linear = {
                    'hidden_from': hidden_states[:, :, head_num, :], 
                    'hidden_to': hidden_states[:, :, head_num, :], 
                    'pos_from': positions,
                    'pos_to': positions,
                    'relev_pos_from': seq_len - positions,
                    'relev_pos_to': seq_len - positions,
                    'inv_pos_from': (positions / seq_len),
                    'inv_pos_to': (positions / seq_len),
                    'inv_relev_pos_from': ((seq_len - positions) / seq_len),
                    'inv_relev_pos_to': ((seq_len - positions) / seq_len),
                    'seq_len': torch.tensor([seq_len]), 
                    'inv_seq_len': (1 / torch.tensor([seq_len])),
                }
                
                data_to_linear = {k:full_data_to_linear[k].to(self.linear_config['device']) for k in self.linear_config['features'] if k != 'head_num'}

                namespace = {'predicted_attention': None, 'self': self, 'data_to_linear': data_to_linear, 'seq_len': seq_len}
                exec(f"predicted_attention = self.linear_model_{head_num}(seq_len, **data_to_linear)", namespace)
                # print(namespace['predicted_attention'])
                attention_probs = nn.Sigmoid()(torch.exp(namespace['predicted_attention']))  # torch.nn.functional.softmax(predicted_attention, dim=-1)
                attentions.append(attention_probs)

            attention_probs = torch.stack(attentions, dim=1)
            context_layer = torch.matmul(attention_probs, value)
    
            attn_output = self._merge_heads(context_layer, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            # print(context_layer.shape, attn_output.shape)
                    
            outputs = (attn_output, None)
            if output_attentions:
                outputs += (attention_probs, )
            # print(attention_probs.shape, context_layer.shape)
            return outputs


        

        else:
            seq_len = hidden_states.shape[1]
            positions = torch.arange(seq_len).view(-1, 1)
            full_data_to_linear = {
                'hidden_from': hidden_states, 
                'hidden_to': hidden_states, 
                'pos_from': positions,
                'pos_to': positions,
                'relev_pos_from': seq_len - positions,
                'relev_pos_to': seq_len - positions,
                'inv_pos_from': (positions / seq_len),
                'inv_pos_to': (positions / seq_len),
                'inv_relev_pos_from': ((seq_len - positions) / seq_len),
                'inv_relev_pos_to': ((seq_len - positions) / seq_len),
                'seq_len': torch.tensor([seq_len]), 
                'inv_seq_len': (1 / torch.tensor([seq_len])),
            }
            if 'head_num' in self.linear_config['features']:
                attentions = []
                for head_num in range(self.linear_config['num_heads']):
                    full_data_to_linear['head_num'] = torch.tensor([head_num])

                    data_to_linear = {k:full_data_to_linear[k].to(self.linear_config['device']) for k in self.linear_config['features']}
                    predicted_attention= self.linear_model(seq_len, **data_to_linear)
                
                    attention_probs = nn.Sigmoid()(torch.exp(predicted_attention))  # torch.nn.functional.softmax(predicted_attention, dim=-1)
                    attentions.append(attention_probs)

                    attention_probs = torch.stack(attentions, dim=1)
                    context_layer = torch.matmul(attention_probs, value.squeeze(1))
                    # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                    
                    # context_layer = context_layer.view(new_context_layer_shape)
            
                    # attn_output = (context_layer, None)
                    # attn_output = torch.vstack(attn_output)
                    # print(context_layer.shape)
                    attn_output = self._merge_heads(context_layer, self.num_heads, self.head_dim)
                    attn_output = self.c_proj(attn_output)
                    attn_output = self.resid_dropout(attn_output)
                    # print(context_layer.shape, attn_output.shape)
                    
                    outputs = (attn_output, None)
                    if output_attentions:
                        outputs += (attention_probs, )
                    # print(attention_probs.shape, context_layer.shape)
                    # print(type(outputs[0]))

                    return outputs
                
            else:
                data_to_linear = {k:full_data_to_linear[k].to(self.linear_config['device']) for k in self.linear_config['features']}
                predicted_attention= self.linear_model(seq_len, **data_to_linear)
            
                attention_probs = nn.Sigmoid()(torch.exp(predicted_attention))  # torch.nn.functional.softmax(predicted_attention, dim=-1)
                context_layer = torch.matmul(attention_probs, value.squeeze(1))
        
                context_layer = context_layer #.permute(0, 2, 1, 3).contiguous()
                #new_context_layer_shape = value.shape[3]
                #attn_output = context_layer.view(new_context_layer_shape)

                attn_output = self._merge_heads(context_layer, self.num_heads, self.head_dim)
                attn_output = self.c_proj(attn_output)
                attn_output = self.resid_dropout(attn_output)
        
                # attn_output = (context_layer, attention_probs) if output_attentions else (context_layer,)
                # attn_output = torch.vstack(attn_output)
                outputs = (attn_output, None)
                if output_attentions:
                    outputs += (attention_probs, )
                # print(attention_probs.shape, attn_output.shape)
                    
                return outputs

class GPTWrapperLin(nn.Module):
    def __init__(self, model, new_attention_class, linear_config, layer_nums=None):
        super().__init__()

        self.gpt2_model = deepcopy(model)
        self.new_attention_class = new_attention_class
        self.layer_nums = layer_nums

        # Create a list of modules to modify
        modules_to_modify = []
        for i in range(12):
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                linear_attention = new_attention_class(self.gpt2_model.config, linear_config) # self.gpt2_model.config, 
                linear_attention.load_state_dict(self.gpt2_model.h[i].attn.state_dict(), strict=False)
                self.gpt2_model.h[i].attn = linear_attention

    def forward(self, *args, **kwargs):
        return self.gpt2_model(*args, **kwargs)
