from transformers import Conv1D
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, \
    BaseModelOutputWithPastAndCrossAttentions
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from copy import deepcopy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import prune_conv1d_layer, find_pruneable_heads_and_indices


class UniformBert2Attention(BertSelfAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            special_tokens_idxs: Optional[List[int]] = [0]
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
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            
        
        mean_hidden_states = torch.mean(value_layer, dim=-2)
        new_mean_hidden_states_shape = mean_hidden_states.size()[:-2] + (self.all_head_size,)
        mean_hidden_states = mean_hidden_states.view(new_mean_hidden_states_shape)

        return mean_hidden_states, None  # Return the mean hidden states

        
class BertWrapperUniform(nn.Module):
    def __init__(self, model, new_attention_class, layer_nums=None, window_size=2):
        super().__init__()

        self.bert_model = deepcopy(model)
        self.layer_nums = layer_nums
        self.new_attention_class = new_attention_class

        # Create a list of modules to modify
        modules_to_modify = []
        for i in range(12):
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                mean_attention = new_attention_class(self.bert_model.config)
                mean_attention.load_state_dict(self.bert_model.encoder.layer[i].attention.self.state_dict())
                self.bert_model.encoder.layer[i].attention.self = mean_attention

    def forward(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)
