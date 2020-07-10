# encoding=utf-8

import re
import torch
import random
import importlib
import numpy as np
from typing import List, Iterable


PADDING = '<pad>'
CODE_PAD = '<pad>'
TGT_START = '<s>'
TGT_END = '</s>'
UNK = '<unk>'

ACTION_2_TGT_ACTION = {
    'insert': '<insert>',
    'delete': '<delete>',
    'replace': '<replace>',
    'equal': '<equal>'
}

FLOAT_TYPE = torch.float


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # set random seed for all devices (both CPU and GPU)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ids_to_input_tensor(word_ids: List[List[int]], pad_token: int, device: torch.device) -> torch.Tensor:
    sents_t = input_transpose(word_ids, pad_token)
    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
    return sents_var


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def get_attr_by_name(class_name: str):
    class_tokens = class_name.split('.')
    assert len(class_tokens) > 1
    module_name = ".".join(class_tokens[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_tokens[-1])


def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]


def recover_desc(sent: Iterable[str]) -> str:
    return re.sub(r' <con> ', "", " ".join(sent))

