#!/usr/bin/env python
"""
Usage:
    vocab.py --train-set=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-set=<file>         Train set file
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
    --use-ft                   build embeddings with fasttext
    --ft-model=<file>          the path of ft pre-trained model [default: cc.en.300.bin]
    --embedding-file=<file>    the file to store built embeddings [default: vocab_embeddings.pkl]
    --vocab-class=<str>        the class name of used Vocab class [default: Vocab]
"""

import os
import pickle
import fasttext
from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING, Union
from collections import Counter
from itertools import chain
from docopt import docopt
import json
from utils.common import *
from tqdm import tqdm
if TYPE_CHECKING:
    from dataset import Dataset


class BaseVocabEntry(ABC):
    def __init__(self):
        self.word2id = None
        self.id2word = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def index2word(self, wid):
        pass

    @abstractmethod
    def indices2words(self, word_ids: List[int]):
        pass

    @abstractmethod
    def words2indices(self, sents: Union[List[str], List[List[str]]]):
        pass


class VocabEntry(BaseVocabEntry):
    def __init__(self, word2id=None):
        super(VocabEntry, self).__init__()
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[PADDING] = 0
            self.word2id[TGT_START] = 1
            self.word2id[TGT_END] = 2
            self.word2id[UNK] = 3

        self.unk_id = self.word2id[UNK]

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = None

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def index2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids) -> List[str]:
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_var = ids_to_input_tensor(word_ids, self[PADDING], device)
        return sents_var

    @staticmethod
    def from_corpus(corpus: Iterable[List[str]], size: int, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry

    def build_ft_embeddings(self, pretrained: fasttext.FastText._FastText):
        embedding_list = []
        for index in tqdm(range(len(self))):
            word = self.index2word(index)
            embedding_list.append(pretrained.get_word_vector(word))
        self.embeddings = np.vstack(embedding_list)
        assert self.embeddings.shape == (len(self), pretrained.get_dimension())

    def build_embeddings(self, pretrained):
        """
        :param pretrained: pre-trained model or vectors
        :return: None
        """
        self.build_ft_embeddings(pretrained)


class ExtVocabEntry(BaseVocabEntry):
    def __init__(self, base_vocab: VocabEntry, sent: List[str]):
        super(ExtVocabEntry, self).__init__()
        self.base_vocab = base_vocab
        self.word2id = {}
        cur_id = len(self.base_vocab)
        for word in set(sent):
            if word not in self.base_vocab:
                self.word2id[word] = cur_id
                cur_id += 1
        self.id2word = {i: word for word, i in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.base_vocab[word])

    def __contains__(self, word):
        return word in self.word2id or word in self.base_vocab

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id) + len(self.base_vocab)

    def __repr__(self):
        return 'Extend Vocabulary[size=%d]' % len(self)

    @property
    def ext_size(self):
        return len(self.word2id)

    def index2word(self, wid):
        if wid in self.id2word:
            return self.id2word[wid]
        return self.base_vocab.index2word(wid)

    def indices2words(self, word_ids: List[int]):
        return [self.index2word(w_id) for w_id in word_ids]

    def words2indices(self, sents: Union[List[str], List[List[str]]]):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]


class BaseVocab(ABC):
    @staticmethod
    @abstractmethod
    def build(dataset: "Dataset", vocab_size: int, freq_cutoff: int):
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass

    @staticmethod
    @abstractmethod
    def load(file_path):
        pass


class Vocab(object):
    def __init__(self, code: VocabEntry, action: VocabEntry, nl: VocabEntry):
        self.code = code
        self.action = action
        self.nl = nl

    def build_embeddings(self, pretrain_path: str):
        pretrain_model = fasttext.load_model(pretrain_path)
        self.code.build_embeddings(pretrain_model)
        self.action.build_embeddings(pretrain_model)
        self.nl.build_embeddings(pretrain_model)

    @staticmethod
    def build(dataset: "Dataset", vocab_size: int, freq_cutoff: int) -> 'Vocab':
        print('initialize code vocabulary..')
        code_vocab = VocabEntry.from_corpus(dataset.get_code_tokens(), vocab_size, freq_cutoff)

        print('initialize action vocabulary..')
        # action only have four values
        action_word2id = {
            PADDING: 0,
            UNK: 1,
            'equal': 2,
            'insert': 3,
            'replace': 4,
            'delete': 5,

        }
        action_vocab = VocabEntry(action_word2id)

        print('initialize nl vocabulary..')
        nl_vocab = VocabEntry.from_corpus(dataset.get_nl_tokens(), vocab_size, freq_cutoff)

        return Vocab(code_vocab, action_vocab, nl_vocab)

    def save(self, file_path: str):
        assert file_path.endswith(".json")
        with open(file_path, 'w') as f:
            json.dump(dict(code_word2id=self.code.word2id,
                           action_word2id=self.action.word2id,
                           nl_word2id=self.nl.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            entry = json.load(f)
        code_word2id = entry['code_word2id']
        action_word2id = entry['action_word2id']
        nl_word2id = entry['nl_word2id']

        return Vocab(VocabEntry(code_word2id), VocabEntry(action_word2id), VocabEntry(nl_word2id))

    def save_embeddings(self, file_path: str):
        embedding_dict = dict(code_embeddings=self.code.embeddings,
                              action_embeddings=self.action.embeddings,
                              nl_embeddings=self.nl.embeddings)
        with open(file_path, 'wb') as f:
            pickle.dump(embedding_dict, f)

    def load_embeddings(self, file_path: str):
        with open(file_path, 'rb') as f:
            embedding_dict = pickle.load(f)
        self.code.embeddings = embedding_dict['code_embeddings']
        self.action.embeddings = embedding_dict['action_embeddings']
        self.nl.embeddings = embedding_dict['nl_embeddings']

    def __repr__(self):
        return 'Vocab(code %d words, action %d words, nl %d words)' % (len(self.code), len(self.action), len(self.nl))


class MixVocab(object):
    def __init__(self, token: VocabEntry, action: VocabEntry):
        self.token = token
        self.action = action

    @staticmethod
    def build(dataset: "Dataset", vocab_size: int, freq_cutoff: int) -> 'MixVocab':
        print('initialize token vocabulary..')
        token_vocab = VocabEntry.from_corpus(dataset.get_mixed_tokens(), vocab_size, freq_cutoff)

        print('initialize action vocabulary..')
        # action only have four values
        action_word2id = {
            PADDING: 0,
            UNK: 1,
            'equal': 2,
            'insert': 3,
            'replace': 4,
            'delete': 5,
        }
        action_vocab = VocabEntry(action_word2id)

        return MixVocab(token_vocab, action_vocab)

    def build_embeddings(self, pretrain_path: str):
        pretrain_model = fasttext.load_model(pretrain_path)
        self.token.build_embeddings(pretrain_model)
        self.action.build_embeddings(pretrain_model)

    def save(self, file_path: str):
        assert file_path.endswith(".json")
        with open(file_path, 'w') as f:
            json.dump(dict(token_word2id=self.token.word2id,
                           action_word2id=self.action.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            entry = json.load(f)
        token_word2id = entry['token_word2id']
        action_word2id = entry['action_word2id']

        return MixVocab(VocabEntry(token_word2id), VocabEntry(action_word2id))

    def save_embeddings(self, file_path: str):
        embedding_dict = dict(token_embeddings=self.token.embeddings,
                              action_embeddings=self.action.embeddings)
        with open(file_path, 'wb') as f:
            pickle.dump(embedding_dict, f)

    def load_embeddings(self, file_path: str):
        with open(file_path, 'rb') as f:
            embedding_dict = pickle.load(f)
        self.token.embeddings = embedding_dict['token_embeddings']
        self.action.embeddings = embedding_dict['action_embeddings']

    def __repr__(self):
        return 'MixVocab(token %d words, action %d words)' % (len(self.token), len(self.action))


if __name__ == '__main__':
    args = docopt(__doc__)
    from dataset import Dataset

    print("Loading train set: " + args['--train-set'])
    train_set = Dataset.create_from_file(args['--train-set'])

    vocab_class = globals()[args['--vocab-class']]
    vocab = vocab_class.build(train_set, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, {}'.format(vocab))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])

    if bool(args['--use-ft']):
        print('Build pre-trained embeddings for each vocab')
        model_path = os.path.expanduser(args['--ft-model'])
        vocab.build_embeddings(model_path)
        vocab.save_embeddings(args['--embedding-file'])
        print('vocabulary embeddings saved to {}'.format(args['--embedding-file']))
