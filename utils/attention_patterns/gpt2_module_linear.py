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
    
class LinearClassifierGPT2Attention(GPT2Attention):

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

        seq_len = hidden_states.shape[1]
        hidden_states_attention_features = self.hidden_pred(hidden_states)
        
        predicted_attention = hidden_states_attention_features.repeat(1, 1, seq_len).transpose(1, 2) # bs, seq_len, 1
        
        positions = torch.arange(seq_len)

        #special tokens
        special_tokens_idxs = (encoder_hidden_states[0] < 103).nonzero().squeeze()
        
        # add i, j pos features
        positions_features_from = positions * self.pos_i_coef # seq_len
        positions_features_out = positions * self.pos_j_coef # seq_len
        predicted_attention += positions_features_from.T
        predicted_attention += positions_features_out
        
        # add relev i, j pos features
        relev_positions_features_from = (seq_len - positions) * self.relev_pos_i_coef # seq_len
        relev_positions_features_out = (seq_len - positions) * self.relev_pos_j_coef # seq_len
        predicted_attention += relev_positions_features_from.T
        predicted_attention += relev_positions_features_out
        
        inv_positions_features_from = (positions / seq_len) * self.inv_pos_i_coef # seq_len
        inv_positions_features_out = (positions / seq_len) * self.inv_pos_j_coef # seq_len
        predicted_attention += inv_positions_features_from.T
        predicted_attention += inv_positions_features_out
        
        inv_relev_positions_features_from = ((seq_len - positions) / seq_len) * self.inv_relev_pos_i_coef # seq_len
        inv_relev_positions_features_out = ((seq_len - positions) / seq_len) * self.inv_relev_pos_j_coef # seq_len
        predicted_attention += inv_relev_positions_features_from.T
        predicted_attention += inv_relev_positions_features_out
        
        seq_len_feature = self.seq_len_coef * seq_len # 1
        inv_seq_len_feature = self.inv_seq_len_coef * (1 / seq_len) # 1
        layer_feature = self.layer * self.layer_coef # 1
        predicted_attention += seq_len_feature
        predicted_attention += inv_seq_len_feature
        predicted_attention += layer_feature

        attention_probs = torch.exp(predicted_attention) #torch.nn.functional.softmax(predicted_attention, dim=-1)
        context_layer = torch.matmul(attention_probs, value)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    


class GPT2WrapperLin(nn.Module):
    def __init__(self, model, new_attention_class, final_model_weights, layer_nums=None, window_size=2):
        super().__init__()

        self.gpt2_model = deepcopy(model)
        self.layer_nums = layer_nums


        # Create a list of modules to modify
        modules_to_modify = []

        for i in layer_nums:
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                mean_attention = new_attention_class(self.gpt2_model.config)
                mean_attention.set_window_size(window_size)
                mean_attention.load_state_dict(self.gpt2_model.h[i].attn.state_dict())
                mean_attention.set_models(final_model_weights, i)
                self.gpt2_model.h[i].attn = mean_attention
           
            

    def forward(self, *args, **kwargs):
        return self.gpt2_model(*args, **kwargs)
