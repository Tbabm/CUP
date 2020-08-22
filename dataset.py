# encoding=utf-8

import math
import json
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Iterable, Tuple, Callable
from utils.common import *
import logging
from vocab import VocabEntry, ExtVocabEntry

logging.basicConfig(level=logging.INFO)


class AbstractExample(ABC):
    @property
    @abstractmethod
    def src_tokens(self):
        pass

    @property
    @abstractmethod
    def tgt_tokens(self):
        pass


class Example(AbstractExample):
    def __init__(self, instance):
        self._sample_id = instance['sample_id']
        self._code_change_seqs = instance['code_change_seq']
        assert self._code_change_seqs is not None
        self._src_desc_tokens = instance['src_desc_tokens']
        # NOTE: add START and END marks in tgt_tokens
        self._tgt_desc_tokens = instance['dst_desc_tokens']
        # for debugging
        self.src_method = instance['src_method']
        self.tgt_method = instance['dst_method']
        self.src_desc = instance['src_desc']
        self.tgt_desc = instance['dst_desc']
        # for pointer generator
        self.src_ext_vocab = None
        self.code_ext_vocab = None
        self.both_ext_vocab = None

    @staticmethod
    def create_partial_example(instance):
        assert 'code_change_seq' in instance
        assert 'src_desc_tokens' in instance
        instance['sample_id'] = 0
        instance['dst_desc_tokens'] = []
        instance['src_method'] = ""
        instance['dst_method'] = ""
        instance['src_desc'] = ""
        instance['dst_desc'] = ""
        return Example(instance)

    @staticmethod
    def create_zero_example():
        instance = {
            'code_change_seq': [[PADDING, PADDING, UNK]],
            'src_desc_tokens': [PADDING, PADDING],
            'dst_desc_tokens': [PADDING, PADDING],
            'src_method': "",
            'dst_method': "",
            'src_desc': "",
            'dst_desc': ""
        }
        return Example(instance)

    @property
    def old_code_tokens(self):
        return [seq[0] for seq in self._code_change_seqs]

    @property
    def new_code_tokens(self):
        return [seq[1] for seq in self._code_change_seqs]

    @property
    def edit_actions(self):
        return [seq[2] for seq in self._code_change_seqs]

    @property
    def code_len(self):
        return len(self._code_change_seqs)

    @property
    def src_tokens(self):
        """
        used for models
        """
        return self._src_desc_tokens

    @property
    def tgt_in_tokens(self):
        return [TGT_START] + self._tgt_desc_tokens

    @property
    def tgt_out_tokens(self):
        return self._tgt_desc_tokens + [TGT_END]

    @property
    def tgt_tokens(self):
        """
        used for models
        """
        return [TGT_START] + self._tgt_desc_tokens + [TGT_END]

    def get_src_desc_tokens(self):
        return self._src_desc_tokens

    def get_tgt_desc_tokens(self):
        return self._tgt_desc_tokens

    def get_code_tokens(self):
        code_tokens = []
        for seq in self._code_change_seqs:
            for token in seq[:2]:
                code_tokens.append(token)
        return code_tokens

    def get_nl_tokens(self):
        """
        used for build vocab
        """
        return self._src_desc_tokens + self._tgt_desc_tokens

    @property
    def tgt_words_num(self):
        return len(self.tgt_tokens) - 1

    def get_src_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.src_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build src_ext_vocab")
            self.src_ext_vocab = ExtVocabEntry(base_vocab, self.src_tokens)
        return self.src_ext_vocab

    def get_code_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.code_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build code_ext_vocab")
            self.code_ext_vocab = ExtVocabEntry(base_vocab, self.new_code_tokens)
        return self.code_ext_vocab

    def get_both_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.both_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build both_ext_vocab")
            # combine the two tokens
            self.both_ext_vocab = ExtVocabEntry(base_vocab, self.src_tokens + self.new_code_tokens)
        return self.both_ext_vocab


class Batch(object):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item) -> Example:
        return self.examples[item]

    @staticmethod
    def create_zero_batch(batch_size: int = 8):
        examples = [Example.create_zero_example() for _ in range(batch_size)]
        return Batch(examples)

    @property
    def tgt_words_num(self) -> int:
        return sum([e.tgt_words_num for e in self.examples])

    @property
    def old_code_tokens(self) -> List[List[str]]:
        return [e.old_code_tokens for e in self.examples]

    @property
    def new_code_tokens(self) -> List[List[str]]:
        return [e.new_code_tokens for e in self.examples]

    @property
    def edit_actions(self) -> List[List[str]]:
        return [e.edit_actions for e in self.examples]

    @property
    def src_tokens(self):
        return [e.src_tokens for e in self.examples]

    @property
    def tgt_in_tokens(self):
        return [e.tgt_in_tokens for e in self.examples]

    @property
    def tgt_out_tokens(self):
        return [e.tgt_out_tokens for e in self.examples]

    @property
    def tgt_tokens(self):
        return [e.tgt_tokens for e in self.examples]

    def get_code_change_tensors(self, code_vocab: VocabEntry, action_vocab: VocabEntry, device: torch.device):
        code_tensor_a = code_vocab.to_input_tensor(self.old_code_tokens, device)
        code_tensor_b = code_vocab.to_input_tensor(self.new_code_tokens, device)

        edit_tensor = action_vocab.to_input_tensor(self.edit_actions, device)

        return code_tensor_a, code_tensor_b, edit_tensor

    def get_src_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.src_tokens, device)

    def get_tgt_in_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.tgt_in_tokens, device)

    def get_tgt_out_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.tgt_out_tokens, device)

    def get_src_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_src_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_code_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_code_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_both_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_both_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_src_lens(self):
        return [len(sent) for sent in self.src_tokens]

    def get_code_lens(self):
        return [e.code_len for e in self.examples]

    def get_src_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tensor:
        word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_src_ext_vocab(base_vocab)
            word_ids.append(ext_vocab.words2indices(e.src_tokens))
        sents_var = ids_to_input_tensor(word_ids, base_vocab[PADDING], device)
        # (src_sent_len, batch_size)
        return sents_var

    def get_code_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tensor:
        """
        :param nl_vocab: the vocab of the generated tokens
        :param device:
        :return:
        """
        word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_code_ext_vocab(base_vocab)
            word_ids.append(ext_vocab.words2indices(e.new_code_tokens))
        sents_var = ids_to_input_tensor(word_ids, base_vocab[PADDING], device)
        # (src_code_len, batch_size)
        return sents_var

    def get_both_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tuple[Tensor, Tensor]:
        src_word_ids = []
        code_word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_both_ext_vocab(base_vocab)
            src_word_ids.append(ext_vocab.words2indices(e.src_tokens))
            code_word_ids.append(ext_vocab.words2indices(e.new_code_tokens))
        src_tensor = ids_to_input_tensor(src_word_ids, base_vocab[PADDING], device)
        code_tensor = ids_to_input_tensor(code_word_ids, base_vocab[PADDING], device)
        return src_tensor, code_tensor

    def get_max_src_ext_size(self) -> int:
        return max([e.get_src_ext_vocab().ext_size for e in self])

    def get_max_code_ext_size(self) -> int:
        return max([e.get_code_ext_vocab().ext_size for e in self])

    def get_max_both_ext_size(self) -> int:
        return max([e.get_both_ext_vocab().ext_size for e in self])


class Dataset(object):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    @staticmethod
    def create_from_file(file_path: str, ExampleClass: Callable = Example):
        examples = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                examples.append(ExampleClass(json.loads(line)))
        logging.info("loading {} samples".format(len(examples)))
        return Dataset(examples)

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def get_code_tokens(self):
        for e in self.examples:
            yield e.get_code_tokens()

    def get_nl_tokens(self):
        for e in self.examples:
            yield e.get_nl_tokens()

    def get_mixed_tokens(self):
        for e in self.examples:
            yield e.get_code_tokens() + e.get_nl_tokens()

    def get_ground_truth(self) -> Iterable[List[str]]:
        for e in self.examples:
            # remove the <s> and </s>
            yield e.get_tgt_desc_tokens()

    def get_src_descs(self) -> Iterable[List[str]]:
        for e in self.examples:
            yield e.get_src_desc_tokens()

    def _batch_iter(self, batch_size: int, shuffle: bool, sort_by_length: bool) -> Batch:
        batch_num = math.ceil(len(self) / batch_size)
        index_array = list(range(len(self)))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [self[idx] for idx in indices]

            if sort_by_length:
                examples = sorted(examples, key=lambda e: len(e.src_tokens), reverse=True)
            yield Batch(examples)

    def train_batch_iter(self, batch_size: int, shuffle: bool) -> Batch:
        for batch in self._batch_iter(batch_size, shuffle=shuffle, sort_by_length=True):
            yield batch

    def infer_batch_iter(self, batch_size):
        for batch in self._batch_iter(batch_size, shuffle=False, sort_by_length=False):
            yield batch
