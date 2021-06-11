# encoding=utf-8

import difflib
from abc import abstractmethod, ABC
from typing import List, Callable, Tuple

from antlr4 import Token

from .javatokenizer.JavaLexer import JavaLexer
from .javatokenizer.tokenizer import tokenize_identifier, tokenize_string_literal, tokenize_java_code_origin


def token_filter_raw(origin_tokens: List[Token], literal_mapping: dict):
    new_tokens = []
    for t in origin_tokens:
        if t.type in [JavaLexer.COMMENT, JavaLexer.LINE_COMMENT, JavaLexer.WS]:
            continue
        elif t.type in list(literal_mapping.keys()):
            t.text = literal_mapping[t.type]
            new_tokens.append(t)
        else:
            # for keywords and identifiers
            new_tokens.append(t)
    return new_tokens


def empty_token_filter(origin_tokens: List[Token]):
    literal_mapping = {}
    return token_filter_raw(origin_tokens, literal_mapping)


class AbstractDiffTokenizer(ABC):
    @abstractmethod
    def tokenize_diff(self, src_method: str, dst_method: str) -> Tuple[List[Token], List[Token]]:
        pass

    def __call__(self, *args, **kwargs):
        return self.tokenize_diff(*args, **kwargs)


class DiffTokenizer(AbstractDiffTokenizer):
    def __init__(self, token_filter: Callable = empty_token_filter):
        self.token_filter = token_filter

    def tokenize_diff(self, src_method: str, dst_method: str) -> Tuple[List[Token], List[Token]]:
        src_method_tokens = tokenize_java_code_origin(src_method)
        dst_method_tokens = tokenize_java_code_origin(dst_method)
        src_method_tokens = self.token_filter(src_method_tokens)
        dst_method_tokens = self.token_filter(dst_method_tokens)
        return src_method_tokens, dst_method_tokens


def _heuristic_replace_match(a_tokens: List[str], b_tokens: List[str]):
    diff_seqs = []
    a_len = len(a_tokens)
    b_len = len(b_tokens)
    delta_len = max(a_len - b_len, b_len - a_len)
    if a_len != b_len:
        # simple situation
        head_ratio = difflib.SequenceMatcher(None, a_tokens[0], b_tokens[0]).quick_ratio()
        tail_ratio = difflib.SequenceMatcher(None, a_tokens[-1], b_tokens[-1]).quick_ratio()
        if head_ratio >= tail_ratio:
            if a_len > b_len:
                b_tokens += [""] * delta_len
            else:
                a_tokens += [""] * delta_len
        else:
            if a_len > b_len:
                b_tokens = [""] * delta_len + b_tokens
            else:
                a_tokens = [""] * delta_len + a_tokens
    assert len(a_tokens) == len(b_tokens)
    for at, bt in zip(a_tokens, b_tokens):
        if at == "":
            diff_seqs.append([at, bt, "insert"])
        elif bt == "":
            diff_seqs.append([at, bt, "delete"])
        else:
            diff_seqs.append([at, bt, "replace"])
    return diff_seqs


def construct_diff_sequence(a: List[str], b: List[str]) -> List[List[str]]:
    diff_seqs = []
    diff = difflib.SequenceMatcher(None, a, b)

    for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
        a_tokens = a[a_i:a_j]
        b_tokens = b[b_i:b_j]
        if op == "delete":
            for at in a_tokens:
                diff_seqs.append([at, "", op])
        elif op == "insert":
            for bt in b_tokens:
                diff_seqs.append(["", bt, op])
        elif op == "equal":
            for at, bt in zip(a_tokens, b_tokens):
                diff_seqs.append([at, bt, op])
        else:
            # replace
            diff_seqs += _heuristic_replace_match(a_tokens, b_tokens)

    return diff_seqs


def construct_diff_sequence_with_con(a: List[Token], b: List[Token]) -> List[List[str]]:
    pre_diff_sequence = construct_diff_sequence([w.text for w in a], [w.text for w in b])

    def _get_sub_tokens(t: Token):
        if t.type == JavaLexer.IDENTIFIER:
            return tokenize_identifier(t.text, with_con=True)
        elif t.type == JavaLexer.STRING_LITERAL:
            return tokenize_string_literal(t.text, with_con=True)
        else:
            return [t.text]

    a_index = 0
    b_index = 0
    new_diff_sequence = []
    for diff in pre_diff_sequence:
        assert diff[0] or diff[1]
        a_token = None
        b_token = None

        if diff[0]:
            a_token = a[a_index]
            a_sub_tokens = _get_sub_tokens(a_token)
            a_index += 1
        else:
            a_sub_tokens = []
        if diff[1]:
            b_token = b[b_index]
            b_sub_tokens = _get_sub_tokens(b_token)
            b_index += 1
        else:
            b_sub_tokens = []

        sub_token_seqs = construct_diff_sequence(a_sub_tokens, b_sub_tokens)
        new_diff_sequence += sub_token_seqs

    return new_diff_sequence
