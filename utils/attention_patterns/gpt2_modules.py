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
