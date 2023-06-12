from transformers import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import prune_conv1d_layer, find_pruneable_heads_and_indices


class MeanGPT2Attention(GPT2Attention):
    # self,

    #     hidden_states: Optional[Tuple[torch.FloatTensor]],
    #     layer_past: Optional[Tuple[torch.Tensor]] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = False,
    #     output_attentions: Optional[bool] = False,
    # ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

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

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, _ = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)

        return outputs

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size, num_heads, seq_len, _ = query.size()

        mean_hidden_states = torch.mean(value, dim=-2)

        # Expand mean_hidden_states to match the shape of attn_output
        expanded_mean_hidden_states = mean_hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)

        return expanded_mean_hidden_states, None  # No attention weights


class GPT2Wrapper(nn.Module):
    def __init__(self, model, attn_class=GPT2Attention):
        super().__init__()

        self.gpt2_model = model

        # Create a dictionary to store the modified modules
        modified_modules = {}

        # Iterate over the modules of the GPT2 model
        for name, module in self.gpt2_model.named_modules():
            if isinstance(module, GPT2Attention):
                # Replace GPT2Attention with MeanGPT2Attention
                modified_modules[name] = MeanGPT2Attention(self.gpt2_model.config)

        # Replace the modules in self.gpt2_model
        for name, module in modified_modules.items():
            setattr(self.gpt2_model, name, module)

    def forward(self, input_ids, attention_mask=None):
        # Forward the inputs through the modified GPT2 model
        return self.gpt2_model(input_ids, attention_mask)


if __name__ == '__main__':
    # Load the pretrained GPT2 model
    gpt2_model = GPT2Model.from_pretrained("gpt2")

    # Create an instance of the wrapper
    wrapper = GPT2Wrapper(gpt2_model)

    # Use the wrapper to process inputs
    inputs = torch.tensor([[1, 2, 3]])
    outputs = wrapper(inputs)
