# encoding=utf-8

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import models.base
from models.base import permute_lstm_output, LSTM


class BaseEncoder(nn.Module, ABC):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.embed_layer = None

    @property
    @abstractmethod
    def output_size(self):
        pass


class Encoder(BaseEncoder):
    def __init__(self, embed_size, hidden_size, embed_layer, dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed_layer = embed_layer

        self.rnn_layer = LSTM(self.embed_size, self.hidden_size, 1, bidirectional=True, batch_first=False,
                              dropout=dropout)

    @property
    def output_size(self):
        return self.hidden_size * 2

    def forward(self, src_tensor: torch.Tensor, src_lens: List[int]):
        """
        :param src_tensor: (src_sent_len, batch_size)
        :param src_lens:
        :return: (batch_size, src_sent_len, self.output_size),
                (num_layers*num_directions, batch, hidden_size),
                 (num_layers*num_directions, batch, hidden_size)
        """
        # (sent_len, batch_size, embed_size)
        embeddings = self.embed_layer(src_tensor)
        encodings, (temp_last_state, temp_last_cell) = self.rnn_layer(embeddings, src_lens, enforce_sorted=False)
        encodings, last_state, last_cell = permute_lstm_output(encodings, temp_last_state, temp_last_cell)
        return encodings, last_state, last_cell


class SeqEditEncoder(BaseEncoder):
    """
    Sequential Edit Encoder

    Input:
        code_change_sequence: List[List[Tuple[old_code_token, new_code_token, action]]]
    Output:
        edit_representation: Tensor: (batch_size, edit_vec_size)
    """

    def __init__(self, embed_size, edit_vec_size, code_embed_layer, action_embed_layer, dropout):
        super(SeqEditEncoder, self).__init__()
        self.edit_vec_size = edit_vec_size
        self.code_embed_layer = code_embed_layer
        self.action_embed_layer = action_embed_layer
        hidden_size = edit_vec_size // 2
        self.rnn_layer = LSTM(embed_size * 3, hidden_size, 1, bidirectional=True, batch_first=False, dropout=dropout)

    @property
    def output_size(self):
        return self.edit_vec_size

    def forward(self, old_token_tensor: Tensor, new_token_tensor: Tensor,
                action_tensor: Tensor, sent_lens: List[int]):
        """
        :param old_token_tensor: (sent_len, batch_size)
        :param new_token_tensor: (sent_len, batch_size)
        :param action_tensor: (sent_len, batch_size)
        :param sent_lens: code sent lens
        :return:
            edit_encodings: (batch_size, sent_len, edit_vec_size)
            last_state: (batch_size, edit_vec_size)
            last_cell: (batch_size, edit_vec_size)
        """
        old_token_embeddings = self.code_embed_layer(old_token_tensor)
        new_token_embeddings = self.code_embed_layer(new_token_tensor)
        action_embeddings = self.action_embed_layer(action_tensor)
        embeddings = torch.cat([old_token_embeddings, new_token_embeddings, action_embeddings], dim=-1)
        encodings, (temp_last_state, temp_last_cell) = self.rnn_layer(embeddings, sent_lens, enforce_sorted=False)
        encodings, last_state, last_cell = models.base.permute_lstm_output(encodings, temp_last_state, temp_last_cell)
        return encodings, last_state, last_cell


class CoAttnLayer(nn.Module):
    def __init__(self, edit_encoding_size, src_encoding_size):
        super().__init__()
        self.edit_src_linear = nn.Linear(edit_encoding_size, src_encoding_size, bias=False)

    def forward(self, edit_encodings: Tensor, src_encodings: Tensor, edit_sent_masks: Tensor, src_sent_masks: Tensor) \
            -> Tuple[Tensor, Tensor]:
        """
        :param edit_encodings: (batch_size, edit_len, edit_encoding_size)
        :param src_encodings: (batch_size, src_len, src_encoding_size)
        :param edit_sent_masks: (batch_size, edit_max_len), **1 for padding**
        :param src_sent_masks: (batch_size, src_max_len), **1 for padding**
        :return: edit_ctx_encodings, src_ctx_encodings
        """
        # similar to dot_prod_attention
        # (batch_size, edit_len, src_len)
        sim_matrix = self.edit_src_linear(edit_encodings).bmm(src_encodings.permute(0, 2, 1))
        # should not mask on the same sim_matrix
        # since softmax on a all-inf column will produce nan
        edit_sim_matrix = sim_matrix.masked_fill(src_sent_masks.unsqueeze(1).bool(), -float('inf'))
        src_sim_matrix = sim_matrix.masked_fill(edit_sent_masks.unsqueeze(-1).bool(), -float('inf'))
        edit_weights = F.softmax(edit_sim_matrix, dim=-1)
        src_weights = F.softmax(src_sim_matrix, dim=1)
        # (batch_size, edit_len, src_encoding_size)
        edit_ctx_encodings = edit_weights.bmm(src_encodings)
        # (batch_size, src_len, edit_encoding_size)
        src_ctx_encodings = src_weights.permute(0, 2, 1).bmm(edit_encodings)
        return edit_ctx_encodings, src_ctx_encodings


class ModelingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.rnn_layer = LSTM(input_size, hidden_size, 1, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, input_tensor: Tensor, sent_lens: List[int]):
        # input is sorted by nl input len, hence enforce_sorted should be False
        # input_tensor: (batch_size, seq_len, input_size)
        encodings, (temp_last_state, temp_last_cell) = self.rnn_layer(input_tensor, sent_lens, enforce_sorted=False)
        # (batch_size, num_layer*num_directions, hidden_size)
        last_state = torch.cat([c.squeeze(0) for c in temp_last_state.split(1, dim=0)], dim=-1)
        last_cell = torch.cat([c.squeeze(0) for c in temp_last_cell.split(1, dim=0)], dim=-1)

        return encodings, last_state, last_cell
