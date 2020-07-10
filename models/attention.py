# encoding=utf-8

from typing import Tuple
import torch
from torch import nn
from torch import Tensor


class SrcAttention(nn.Module):
    def __init__(self, att_func, src_out_size, hidden_size):
        super().__init__()
        self.att_func = att_func
        self.att_vec_linear = nn.Linear(src_out_size + hidden_size, hidden_size, bias=False)

    def forward(self, h_t: Tensor, src_encodings: Tensor, src_encoding_att_linear: Tensor,
                       sent_masks: Tensor = None) -> Tuple[Tensor, Tensor]:
        # ctx_t: src_encoding_state
        ctx_t, alpha_t = self.att_func(h_t, src_encodings, src_encoding_att_linear, sent_masks)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([ctx_t, h_t], 1)))
        return att_t, alpha_t


class BothAttention(nn.Module):
    def __init__(self, att_func, src_out_size, edit_vec_size, hidden_size):
        super().__init__()
        self.att_func = att_func
        self.att_vec_linear = nn.Linear(src_out_size + edit_vec_size + hidden_size, hidden_size, bias=False)

    def forward(self, h_t: Tensor, src_encodings: Tensor, src_encodings_att_linear: Tensor, edit_encodings: Tensor,
                edit_encodings_att_linear: Tensor, src_sent_masks: Tensor, edit_sent_masks: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        src_ctx_t, src_alpha_t = self.att_func(h_t, src_encodings, src_encodings_att_linear, src_sent_masks)
        edit_ctx_t, edit_alpha_t = self.att_func(h_t, edit_encodings, edit_encodings_att_linear, edit_sent_masks)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([src_ctx_t, edit_ctx_t, h_t], 1)))  # E.q. (5)
        return att_t, src_alpha_t, edit_alpha_t
