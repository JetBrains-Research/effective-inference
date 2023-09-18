from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
# from progressbar import progressbar
from tqdm.auto import tqdm
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, \
    BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union, List
from torch import nn
from copy import deepcopy
import joblib
from torch import nn
from copy import deepcopy

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
            
            for arg_name, arg_value in kwargs.items():
                namespace = {'cur_result': None, 'self': self, 'arg_name': arg_name, 'arg_value': arg_value}
                if 'hidden' in arg_name:
                    exec(f"cur_result = self.{arg_name}(arg_value)", namespace)
                else:
                    exec(f"cur_result = self.{arg_name} * arg_value", namespace)
                if 'from' in arg_name:
                    result += namespace['cur_result'].T
                else:
                    result += namespace['cur_result']
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


from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, \
    BaseModelOutputWithPastAndCrossAttentions

class LinearClassifierBertAttention(BertSelfAttention):
    """
    Idea: attention weights are predicted by Linear Classifier
    """
    def __init__(self, bert_config, config):
        super(LinearClassifierBertAttention, self).__init__(bert_config)
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
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            # special_tokens_idxs: Optional[List[int]] = [0]
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
            # special_tokens_idxs = (encoder_hidden_states[0] < 103).nonzero().squeeze()
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
            # special_tokens_idxs = (encoder_hidden_states[0] < 103).nonzero().squeeze()
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

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
            context_layer = torch.matmul(attention_probs, value_layer)
    
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
    
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    
            if self.is_decoder:
                outputs = outputs + (past_key_value,)
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
                    context_layer = torch.matmul(attention_probs, value_layer.squeeze(1))
                    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
                    context_layer = context_layer.view(new_context_layer_shape)
            
                    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            
                    if self.is_decoder:
                        outputs = outputs + (past_key_value,)
                    return outputs
                
            else:
                data_to_linear = {k:full_data_to_linear[k].to(self.linear_config['device']) for k in self.linear_config['features']}
                predicted_attention= self.linear_model(seq_len, **data_to_linear)
            
                attention_probs = nn.Sigmoid()(torch.exp(predicted_attention))  # torch.nn.functional.softmax(predicted_attention, dim=-1)
                context_layer = torch.matmul(attention_probs, value_layer.squeeze(1))
        
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
                context_layer = context_layer.view(new_context_layer_shape)
        
                outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
                if self.is_decoder:
                    outputs = outputs + (past_key_value,)
        return outputs

class BertWrapperLin(nn.Module):
    def __init__(self, model, new_attention_class, linear_config, layer_nums=None, window_size=2):
        super().__init__()

        self.bert_model = deepcopy(model)
        self.layer_nums = layer_nums

        # Create a list of modules to modify
        modules_to_modify = []
        for i in range(len(self.bert_model.encoder.layer)):
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                linear_attention = new_attention_class(self.bert_model.config, linear_config) # self.bert_model.config, 
                linear_attention.load_state_dict(self.bert_model.encoder.layer[i].attention.self.state_dict(), strict=False)

                self.bert_model.encoder.layer[i].attention.self = linear_attention

    def forward(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)