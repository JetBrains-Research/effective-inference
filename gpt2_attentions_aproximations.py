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

        mean_hidden_states = torch.mean(hidden_states, dim=-2)
        expanded_mean_hidden_states = mean_hidden_states.unsqueeze(1)

        if attention_mask is not None:
            # Apply attention mask
            expanded_mean_hidden_states = expanded_mean_hidden_states * attention_mask.unsqueeze(-1)

        return expanded_mean_hidden_states, None  # Return the mean hidden states


class GPT2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.gpt2_model = model

        # Create a list of modules to modify
        modules_to_modify = []
        for name, module in self.gpt2_model.named_modules():
            if isinstance(module, GPT2Attention):
                modules_to_modify.append((name, module))

        # Replace GPT2Attention with MeanGPT2Attention
        for name, module in modules_to_modify:
            mean_attention = MeanGPT2Attention(self.gpt2_model.config)
            mean_attention.load_state_dict(module.state_dict())
            setattr(self.gpt2_model, name, mean_attention)

    def forward(self, *args, **kwargs):
        return self.gpt2_model(*args, **kwargs)
