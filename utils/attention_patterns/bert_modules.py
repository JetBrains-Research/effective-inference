from transformers import Conv1D
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, BaseModelOutputWithPastAndCrossAttentions
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


class WindowBert2Attention(BertSelfAttention):
    
    def set_window_size(self, window_size):
        self.WINDOW_SIZE = window_size
        
    """
    Idea: only a window elements do matter
    Implementation: insted of (n, d) x (d, n) matmul we have (n, d) x (d, k) matmul 
    That means we need to cut Q and V matricies by taking k-sized window. 
    """
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

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        use_cache = past_key_value is not None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        
        attn_weights_store, res_store = [], []
        #print(query_layer.shape, value_layer.shape)
        TOTAL_MAX_SIZE = len(self.special_tokens) + 2 * self.WINDOW_SIZE + 1
        for i in range(query_layer.shape[2]):
            
            l, r = max(0, i-self.WINDOW_SIZE), min(query_layer.shape[2] - 1, i+self.WINDOW_SIZE+1)
            window = list(range(l, r))
            
            for special_token_idx in self.special_tokens:
                if special_token_idx not in window:
                    window.append(special_token_idx)
            # print(i, window) #print(len(window))
            
            #print(window)
            #print(attention_scores.shape)
            attn_weights_i = nn.functional.softmax(attention_scores[:, :, i, window], dim=-1)
            attn_weights_i = attn_weights_i.view(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, attn_weights_i.shape[2])
            #print(attn_weights_i.shape)
            window_values = value_layer[:, :, window, :]
            #print(window_values.shape)
            attn_weights_i_out = torch.matmul(attn_weights_i, window_values)
            #print('-->', attn_weights_i_out.shape)
            
            res_store.append(attn_weights_i_out)
            attn_weights_i_new = torch.zeros(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, TOTAL_MAX_SIZE).to(attn_weights_i.device)
            #print(attn_weights_i_new.shape, attn_weights_i.shape)
            attn_weights_i_new[:, :, :, list(range(len(window)))] = attn_weights_i
            attn_weights_store.append(
                attn_weights_i_new
            )
        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        #print(len(attn_weights_store), attn_weights_store[0].shape, attn_weights_store[-1].shape) 
        #print(len(res_store), res_store[0].shape, res_store[-1].shape) 
        
        attention_probs = torch.cat(attn_weights_store, dim=2)
        context_layer = torch.cat(res_store, dim=2)
        
        #print(attention_probs.shape)
        #print(context_layer.shape)
        

        #context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    
class LinearClassifierBertAttention(BertSelfAttention):
    
    def set_window_size(self, window_size):
        self.WINDOW_SIZE = window_size
        
    def set_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        
    """
    Idea: only a window elements do matter
    Implementation: insted of (n, d) x (d, n) matmul we have (n, d) x (d, k) matmul 
    That means we need to cut Q and V matricies by taking k-sized window. 
    """
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

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        use_cache = past_key_value is not None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        
        attn_weights_store, res_store = [], []
        #print(query_layer.shape, value_layer.shape)
        TOTAL_MAX_SIZE = len(self.special_tokens) + 2 * self.WINDOW_SIZE + 1
        for i in range(query_layer.shape[2]):
            
            l, r = max(0, i-self.WINDOW_SIZE), min(query_layer.shape[2] - 1, i+self.WINDOW_SIZE+1)
            window = list(range(l, r))
            
            for special_token_idx in self.special_tokens:
                if special_token_idx not in window:
                    window.append(special_token_idx)
            # print(i, window) #print(len(window))
            
            #print(window)
            #print(attention_scores.shape)
            attn_weights_i = nn.functional.softmax(attention_scores[:, :, i, window], dim=-1)
            attn_weights_i = attn_weights_i.view(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, attn_weights_i.shape[2])
            #print(attn_weights_i.shape)
            window_values = value_layer[:, :, window, :]
            #print(window_values.shape)
            attn_weights_i_out = torch.matmul(attn_weights_i, window_values)
            #print('-->', attn_weights_i_out.shape)
            
            res_store.append(attn_weights_i_out)
            attn_weights_i_new = torch.zeros(attn_weights_i.shape[0], attn_weights_i.shape[1], 1, TOTAL_MAX_SIZE).to(attn_weights_i.device)
            #print(attn_weights_i_new.shape, attn_weights_i.shape)
            attn_weights_i_new[:, :, :, list(range(len(window)))] = attn_weights_i
            attn_weights_store.append(
                attn_weights_i_new
            )
        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        #print(len(attn_weights_store), attn_weights_store[0].shape, attn_weights_store[-1].shape) 
        #print(len(res_store), res_store[0].shape, res_store[-1].shape) 
        
        attention_probs = torch.cat(attn_weights_store, dim=2)
        context_layer = torch.cat(res_store, dim=2)
        
        #print(attention_probs.shape)
        #print(context_layer.shape)
        

        #context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    
class BertWrapper(nn.Module):
    def __init__(self, model, new_attention_class, layer_nums=None, window_size=2):
        super().__init__()

        self.bert_model = deepcopy(model)
        self.layer_nums = layer_nums
        self.new_attention_class = new_attention_class
        self.window_size = window_size

        # Create a list of modules to modify
        modules_to_modify = []
        for i in range(12):
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                mean_attention = new_attention_class(self.bert_model.config)
                mean_attention.set_window_size(window_size)
                mean_attention.load_state_dict(self.bert_model.encoder.layer[i].attention.self.state_dict())
                self.bert_model.encoder.layer[i].attention.self = mean_attention

    def forward(self, special_tokens_idxs, *args, **kwargs):
        for i in range(12):
            if (self.layer_nums is not None and i in self.layer_nums) or (self.layer_nums is None):
                self.bert_model.encoder.layer[i].attention.self.set_special_tokens(special_tokens_idxs)
        return self.bert_model(*args, **kwargs, special_tokens_idxs=special_tokens_idxs)
