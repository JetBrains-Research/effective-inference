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

WINDOW_SIZE = 5

class MeanGPT2Attention(GPT2Attention):
    
    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

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

        attn_output = torch.mean(value, 1).repeat(1, value.shape[1], 1)
        outputs = (attn_output, None)
        if output_attentions:
            outputs += (None,)

        return outputs # Return the mean hidden states
    
class WindowMEANGPT2Attention(GPT2Attention):
    
    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

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

        attn_output = []
        for batch_el in query:
            for i in range(len(batch_el)):
                #print(batch_el.shape)
                l, r = max(0, i-WINDOW_SIZE), min(batch_el.shape[0], i+WINDOW_SIZE)
                new_embedding = torch.mean(batch_el[l:r], dim=0)
                #print(batch_el[l:r])
                #print(l, r)
                attn_output.append(new_embedding)
        
        #print(attn_output[0].shape)
        #print(attn_output[0])
        attn_output = torch.vstack(attn_output)
        outputs = (attn_output, None)
        if output_attentions:
            outputs += (None,)

        return outputs # Return the mean hidden states
    
class WindowGPT2Attention(GPT2Attention):
    """
    Idea: only a window elements do matter
    Implementation: insted of (n, d) x (d, n) matmul we have (n, d) x (d, k) matmul 
    That means we need to cut Q and V matricies by taking k-sized window. 
    """
    
    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

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

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        #print(attn_output.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        #print(attn_output[0].shape)
        #print(attn_output[0])
        #attn_output = torch.vstack(attn_output)
        outputs = (attn_output, None)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs # Return the mean hidden states
    
    '''
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        WINDOW_SIZE = 5

        #key = key[:, :, [0] + list(range(-WINDOW_SIZE, 1)), :]
        attn_weights_store, res_store = [], []
        for i in range(query.shape[2]):
            q_vec = query[:, :, i, :].view(query.shape[0], query.shape[1], 1, query.shape[3])
            l, r = max(0, i-WINDOW_SIZE+1), i
            window = list(range(l, r+1))
            if 0 in window:
                k_vec = key[:, :, window, :]
            else:
                l, r = max(0, i-WINDOW_SIZE+1), i
                window = [0] + list(range(l, r+1))
                k_vec = key[:, :, window, :]
                # window = [0] + window
            #print(k_vec.shape, q_vec.shape)
            attn_weights_i = torch.zeros(query.shape[0], query.shape[1], 1, 1+WINDOW_SIZE).to('cuda')
            res = torch.matmul(q_vec, k_vec.transpose(-1, -2))
            #print(res.shape)

            res_out = torch.nn.functional.softmax(res, dim=-1)
            #print(res_out.shape)
            window_values = value[:, :, window, :]
            attn_weights_i_out = torch.matmul(res_out, window_values)
            
            attn_weights_i[:, :, :, list(range(len(window)))] = res_out
            attn_weights_store.append(attn_weights_i)
            #print(attn_weights_i_out.shape)
            res_store.append(attn_weights_i_out)

        attn_weights = torch.cat(attn_weights_store, dim=2)
        attn_output = torch.cat(res_store, dim=2)

        return attn_output, attn_weights
    '''
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        #print(value.shape)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        
        attn_weights_store, res_store = [], []
        #print(query.shape[2])
        for i in range(query.shape[2]):
            
            l, r = max(0, i-WINDOW_SIZE+1), i
            window = list(range(l, r+1))
            if 0 not in window:
                window.append(0)
            
            attn_weights_i = nn.functional.softmax(attn_weights[:, :, i, window], dim=-1)
            attn_weights_i = attn_weights_i.view(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, attn_weights_i.shape[2])
            #print(attn_weights_i.shape)
            window_values = value[:, :, window, :]
            #print(window_values.shape)
            attn_weights_i_out = torch.matmul(attn_weights_i, window_values)
            #print('-->', attn_weights_i_out.shape)
            
            res_store.append(attn_weights_i_out)
            attn_weights_i_new = torch.zeros(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, WINDOW_SIZE+1).to(attn_weights_i.device)
            #print(attn_weights_i_new.shape, attn_weights_i.shape)
            attn_weights_i_new[:, :, :, list(range(len(window)))] = attn_weights_i
            attn_weights_store.append(
                attn_weights_i_new
            )
        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        #print(len(attn_weights_store), attn_weights_store[0].shape, attn_weights_store[-1].shape) 
        #print(len(res_store), res_store[0].shape, res_store[-1].shape) 
        
        attn_weights = torch.cat(attn_weights_store, dim=2)
        attn_output = torch.cat(res_store, dim=2)
        
        #print(attn_weights.shape)
        #print(attn_output.shape)

        return attn_output, attn_weights


class GPT2Wrapper(nn.Module):
    def __init__(self, model, new_attention_class):
        super().__init__()

        self.gpt2_model = deepcopy(model)

        # Create a list of modules to modify
        modules_to_modify = []
        for i in range(12):
            mean_attention = new_attention_class(self.gpt2_model.config)
            mean_attention.load_state_dict(self.gpt2_model.h[i].attn.state_dict())
            self.gpt2_model.h[i].attn = mean_attention
            

    def forward(self, *args, **kwargs):
        return self.gpt2_model(*args, **kwargs)
