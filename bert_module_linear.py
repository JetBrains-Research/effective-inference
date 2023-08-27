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

WINDOW_SIZE = 10


class LinearClassifierBertAttention(BertSelfAttention):
    """
    Idea: attention weights are predicted by Linear Classifier
    """

    def set_models(self, coefs, layer):
        # print(coefs.size)
        # print(coefs[768])
        self.hidden_pred = torch.nn.Linear(
            in_features=768,
            out_features=1
        )
        self.hidden_pred.data = torch.tensor(coefs[:768]).view(1, -1).to(DEVICE)
        self.pos_i_coef = coefs[768]
        self.pos_j_coef = coefs[769]
        self.relev_pos_i_coef = coefs[770]
        self.relev_pos_j_coef = coefs[771]
        self.inv_pos_i_coef = coefs[772]
        self.inv_pos_j_coef = coefs[773]
        self.inv_relev_pos_i_coef = coefs[774]
        self.inv_relev_pos_j_coef = coefs[775]
        self.seq_len_coef = coefs[776]
        self.inv_seq_len_coef = coefs[777]
        self.layer_coef = coefs[778]
        self.layer = layer

    def set_window_size(self, window_size):
        self.WINDOW_SIZE = window_size

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
            special_tokens_idxs = (encoder_hidden_states[0] < 103).nonzero().squeeze()
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
            special_tokens_idxs = (encoder_hidden_states[0] < 103).nonzero().squeeze()
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:

            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        seq_len = hidden_states.shape[1]
        hidden_states_attention_features = self.hidden_pred(hidden_states)

        predicted_attention = hidden_states_attention_features.repeat(1, 1,
                                                                      seq_len)  # .transpose(1, 2) # bs, seq_len, 1

        positions = torch.arange(seq_len)

        # add i, j pos features
        positions_features_from = positions * self.pos_i_coef  # seq_len
        positions_features_out = positions * self.pos_j_coef  # seq_len
        predicted_attention += positions_features_from.T
        predicted_attention += positions_features_out

        # add relev i, j pos features
        relev_positions_features_from = (seq_len - positions) * self.relev_pos_i_coef  # seq_len
        relev_positions_features_out = (seq_len - positions) * self.relev_pos_j_coef  # seq_len
        predicted_attention += relev_positions_features_from.T
        predicted_attention += relev_positions_features_out

        inv_positions_features_from = (positions / seq_len) * self.inv_pos_i_coef  # seq_len
        inv_positions_features_out = (positions / seq_len) * self.inv_pos_j_coef  # seq_len
        predicted_attention += inv_positions_features_from.T
        predicted_attention += inv_positions_features_out

        inv_relev_positions_features_from = ((seq_len - positions) / seq_len) * self.inv_relev_pos_i_coef  # seq_len
        inv_relev_positions_features_out = ((seq_len - positions) / seq_len) * self.inv_relev_pos_j_coef  # seq_len
        predicted_attention += inv_relev_positions_features_from.T
        predicted_attention += inv_relev_positions_features_out

        seq_len_feature = self.seq_len_coef * seq_len  # 1
        inv_seq_len_feature = self.inv_seq_len_coef * (1 / seq_len)  # 1
        layer_feature = self.layer * self.layer_coef  # 1
        predicted_attention += seq_len_feature
        predicted_attention += inv_seq_len_feature
        predicted_attention += layer_feature
        attention_probs = torch.exp(predicted_attention)  # torch.nn.functional.softmax(predicted_attention, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer.squeeze(1))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertWrapperLin(nn.Module):
    def __init__(self, model, new_attention_class, final_model_weights_dict, layer_nums=None, window_size=2):
        super().__init__()

        self.bert_model = deepcopy(model)
        self.layer_nums = layer_nums
        self.final_model_weights_dict = final_model_weights_dict

        # Create a list of modules to modify
        modules_to_modify = []
        for i in final_model_weights_dict.keys():
            if (layer_nums is not None and i in layer_nums) or (layer_nums is None):
                mean_attention = new_attention_class(self.bert_model.config)
                mean_attention.set_window_size(window_size)
                mean_attention.load_state_dict(self.bert_model.encoder.layer[i].attention.self.state_dict())
                if final_model_weights_dict is not None:  # and i in final_model_weights_dict.keys():
                    mean_attention.set_models(final_model_weights_dict[i], i)
                self.bert_model.encoder.layer[i].attention.self = mean_attention

    def forward(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)