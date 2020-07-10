# encoding=utf-8
from typing import List, Union

import os
import math
import fire
import json
import psutil
import logging
import difflib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataset import Dataset
from utils.diff_processor import DiffProcessor
from models.beam import Hypothesis

logging.basicConfig(level=logging.INFO)
PROCESS = psutil.Process(os.getpid())


def get_diff(src: str, dst: str):
    diff_lines = list(difflib.unified_diff(src.splitlines(keepends=True), dst.splitlines(keepends=True)))
    return "".join(diff_lines[2:])


def get_preprocessed_diff(src: str, dst: str):
    # preprocess, tokenize and join
    diff = get_diff(src, dst)
    return DiffProcessor().process(diff)


def cal_sim_matrix(train_texts: List[str], test_texts: List[str], vectorizer):
    logging.debug("1. Current memory usage: {}".format(PROCESS.memory_info().rss))
    train_matrix = vectorizer.fit_transform(train_texts)
    logging.info("vocabulary size: {}".format(len(vectorizer.vocabulary_)))
    logging.debug("2. Current memory usage: {}".format(PROCESS.memory_info().rss))
    test_matrix = vectorizer.transform(test_texts)
    logging.debug("3. Current memory usage: {}".format(PROCESS.memory_info().rss))
    test_train_sims = cosine_similarity(test_matrix, train_matrix)
    logging.debug("4. Current memory usage: {}".format(PROCESS.memory_info().rss))
    return test_train_sims


class NNUpdater(object):
    def __init__(self, diff_vectorizer: Union[CountVectorizer, TfidfVectorizer],
                 desc_vectorizer: Union[CountVectorizer, TfidfVectorizer], alpha: float = 0.5):
        self.diff_vectorizer = diff_vectorizer
        self.desc_vectorizer = desc_vectorizer
        self.alpha = alpha

        self._train_diffs = None
        self._test_diffs = None
        self._train_src_descs = None
        self._train_tgt_descs = None
        self._test_src_descs= None

    def get_diffs(self, dataset: Dataset) -> List[str]:
        return [get_preprocessed_diff(e.src_method, e.tgt_method) for e in dataset]

    def get_src_descs(self, dataset: Dataset) -> List[str]:
        return [" ".join(e.get_src_desc_tokens()) for e in dataset]

    def get_tgt_descs(self, dataset: Dataset) -> List[List[str]]:
        # excluding <s> and </s>
        return list(dataset.get_ground_truth())

    def train(self, train_set):
        self._train_diffs = self.get_diffs(train_set)
        self._train_src_descs = self.get_src_descs(train_set)
        self._train_tgt_descs = self.get_tgt_descs(train_set)

    def cal_similarity(self, train_texts: List[str], test_texts: List[str], vectorizer):
        return cal_sim_matrix(train_texts, test_texts, vectorizer)

    def infer(self, test_set: Dataset) -> List[List[Hypothesis]]:
        assert self._train_diffs is not None
        assert self._train_src_descs is not None

        self._test_diffs = self.get_diffs(test_set)
        self._test_src_descs = self.get_src_descs(test_set)
        ref_descs = self.get_tgt_descs(test_set)

        diff_test_train_sims = self.cal_similarity(self._train_diffs, self._test_diffs, self.diff_vectorizer)
        desc_test_train_sims = self.cal_similarity(self._train_src_descs, self._test_src_descs, self.desc_vectorizer)
        test_train_sims = self.alpha * diff_test_train_sims + (1 - self.alpha) * desc_test_train_sims

        # stable for small index first
        top_k_trains = np.argsort(-test_train_sims, axis=-1, kind='stable')[:, :5]
        assert top_k_trains.ndim == 2 and len(top_k_trains) == len(test_set)

        results = []
        for test_index, train_indexes in enumerate(top_k_trains):
            # the result of each example should be a list of hypothesis
            results.append([Hypothesis(value=self._train_tgt_descs[tid],
                                       score=test_train_sims[test_index][tid]) for tid in train_indexes])

        # for debug
        log_index = 1
        log_train_index = top_k_trains[1][0]
        logging.debug("TEST DIFF:        {}".format(self._test_diffs[log_index]))
        logging.debug("BEST TRAIN DIFF:  {}".format(self._train_diffs[log_train_index]))
        logging.debug("TEST SDESC:       {}".format(self._test_src_descs[log_index]))
        logging.debug("BEST TRAIN SDESC: {}".format(self._train_src_descs[log_train_index]))
        logging.debug("BEST TRAIN DDESC: {}".format(str(self._train_tgt_descs[log_train_index])))
        logging.debug("RESULT:           {}".format(str(results[log_index][0])))
        logging.debug("Reference:        {}".format(str(ref_descs[log_index])))

        return results


def main(train_file="../../data/little_dataset/train.jsonl",
         test_file="../../data/little_dataset/test.jsonl",
         out_file="../../data/little_dataset/nnupdater_result.json",
         alpha=0.5):
    diff_vectorizer = TfidfVectorizer()
    desc_vectorizer = TfidfVectorizer()
    updater = NNUpdater(diff_vectorizer, desc_vectorizer, alpha)

    train_set = Dataset.create_from_file(train_file)
    test_set = Dataset.create_from_file(test_file)
    updater.train(train_set)
    results = updater.infer(test_set)
    with open(out_file, 'w') as f:
        json.dump(results, f)


def slice_test_main(train_file="../../data/little_dataset/train.jsonl",
                    test_file="../../data/little_dataset/test.jsonl",
                    out_file="../../data/little_dataset/nnupdater_result.json",
                    alpha=0.5,
                    slice_count=4,
                    diff_vec_class=None,
                    desc_vec_class=None):
    diff_vectorizer = diff_vec_class() if diff_vec_class else TfidfVectorizer()
    desc_vectorizer = desc_vec_class() if desc_vec_class else TfidfVectorizer()
    logging.info("Diff vectorizer: {}".format(type(diff_vectorizer)))
    logging.info("Desc vectorizer: {}".format(type(desc_vectorizer)))
    logging.info("Alpha: {}".format(alpha))
    if not desc_vectorizer:
        desc_vectorizer = TfidfVectorizer()
    updater = NNUpdater(diff_vectorizer, desc_vectorizer, alpha)

    train_set = Dataset.create_from_file(train_file)
    test_set = Dataset.create_from_file(test_file)
    updater.train(train_set)

    results = []
    slice_size = math.ceil(len(test_set) / slice_count)
    for index in range(slice_count):
        cur_test_set = Dataset(test_set[index*slice_size : (index+1)*slice_size])
        logging.info("Current test set size: {}".format(len(cur_test_set)))
        results += updater.infer(cur_test_set)
    assert len(results) == len(test_set)
    with open(out_file, 'w') as f:
        json.dump(results, f)
    return results


if __name__ == '__main__':
    fire.Fire({
        "main": main,
        "slice_test_main": slice_test_main
    })

