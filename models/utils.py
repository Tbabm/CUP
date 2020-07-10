from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def dot_prod_attention(h_t: Tensor, src_encodings: Tensor, src_encoding_att_linear: Tensor,
                       mask: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    :param h_t: (batch_size, hidden_state)
    :param src_encodings: (batch_size, src_sent_len, src_output_size)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_state)
    :param mask: (batch_size, src_sent_len), paddings are marked as 1
    :return:
        ctx_vec: (batch_size, src_output_size)
        softmaxed_att_weight: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

    if mask is not None:
        att_weight.data.masked_fill_(mask.bool(), -float('inf'))

    softmaxed_att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encodings).squeeze(1)

    return ctx_vec, softmaxed_att_weight


def negative_log_likelihood(logits: torch.FloatTensor, gold_tensor: torch.LongTensor,
                            words_mask: torch.FloatTensor) -> Tensor:
    """
    :param logits: ( batch_size, tgt_vocab_size), log_softmax
    :param gold_tensor: ([tgt_src_len - 1, x], batch_size)
    :param words_mask: ([tgt_src_len - 1, x], batch_size), a matrix to mask target words, 1.0 for non-pad
                       NOTE: this mask is different from dot-production mask
    :return: losses: ([tgt_src_len - 1, x], batch_size)
    """
    # (sent_len, batch_size)
    gold_words_log_prob = torch.gather(logits, index=gold_tensor.unsqueeze(-1), dim=-1).squeeze(-1) * words_mask
    return -gold_words_log_prob
