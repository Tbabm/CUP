# encoding=utf-8

"""
Evaluate results and calculate metrics

Usage:
    eval.py [options] TEST_SET RESULT_FILE

Options:
    -h --help                   show this screen.
    --metrics=<arg...>          metrics to calculate [default: accuracy,recall,distance,nlg]
    --eval-class=<str>          the class used to evaluate [default: Evaluator]
"""
import json
import logging
import stanfordnlp
from utils.common import *
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple
from docopt import docopt
from dataset import Dataset
from utils.common import word_level_edit_distance
from utils.tokenizer import Tokenizer
from nlgeval import NLGEval
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
EMPTY_TOKEN = '<empty>'


class BaseMetric(ABC):
    @abstractmethod
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> float:
        """
        :param hypos: each hypo contains k sents, for accuracy, only use the first sent, for recall, use k sents
        :param references: the dst desc sents
        :param src_references: the src desc sents
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def is_equal(hypo: List[str], ref: List[str]):
        if hypo == ref:
            return True
        if ref[-1] in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_'.split() and ref[:-1] == hypo:
            return True
        return False


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__()
        self.correct_count = 0

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> dict:
        total = 0
        correct = 0
        for hypo_list, ref in zip(hypos, references):
            hypo = hypo_list[0]
            if not hypo:
                hypo = [EMPTY_TOKEN]
            assert (type(hypo[0]) == str)
            assert (type(ref[0]) == str)
            total += 1
            if self.is_equal(hypo, ref):
                correct += 1
        return {'accuracy': correct / total, 'correct_count': correct}


class Recall(BaseMetric):
    def __init__(self, k: int = 5, *args, **kwargs):
        super(Recall, self).__init__()
        self.k = k

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> float:
        total = 0
        correct = 0
        for hypo_list, ref in zip(hypos, references):
            total += 1
            for hypo in hypo_list:
                if self.is_equal(hypo, ref):
                    correct += 1
                    break
        return correct / total


class EditDistance(BaseMetric):
    def __init__(self, *args, **kwargs):
        super(EditDistance, self).__init__()

    @staticmethod
    def edit_distance(sent1: List[str], sent2: List[str]) -> int:
        return word_level_edit_distance(sent1, sent2)

    @classmethod
    def relative_distance(cls, src_ref_dis, hypo_ref_dis):
        if src_ref_dis == 0:
            logging.error("src_ref is the same as ref.")
            src_ref_dis = 1
        return hypo_ref_dis / src_ref_dis

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> dict:
        src_distances = []
        hypo_distances = []
        rel_distances = []
        for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
            hypo = hypo_list[0]
            hypo_ref_dis = self.edit_distance(hypo, ref)
            src_ref_dis = self.edit_distance(src_ref, ref)
            src_distances.append(src_ref_dis)
            hypo_distances.append(hypo_ref_dis)
            rel_distances.append(self.relative_distance(src_ref_dis, hypo_ref_dis))
        rel_dis = float(np.mean(rel_distances))
        src_dis = float(np.mean(src_distances))
        hypo_dis = float(np.mean(hypo_distances))
        # return float(np.mean(distances))
        return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis}


class NLGMetrics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

    @staticmethod
    def prepare_sent(tokens: List[str]) -> str:
        return recover_desc(tokens)

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> dict:
        # List[str]
        first_hypos = [self.prepare_sent(hypo_list[0]) for hypo_list in hypos]
        src_ref_strs = [self.prepare_sent(src_ref) for src_ref in src_references]
        # List[List[str]]
        references_lists = [[self.prepare_sent(ref) for ref in references]]
        # distinct
        metrics_dict = self.nlgeval.compute_metrics(references_lists, first_hypos)
        # relative improve
        src_metrics_dict = self.nlgeval.compute_metrics(references_lists, src_ref_strs)
        relative_metrics_dict = OrderedDict({})
        for key in metrics_dict:
            relative_metrics_dict[key] = (metrics_dict[key] - src_metrics_dict[key]) / src_metrics_dict[key]
        return {
            'Bleu_4': metrics_dict['Bleu_4'],
            'METEOR': metrics_dict['METEOR']
        }


class StanfordNLPTool:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

    def lemmatize(self, sent: List[str]):
        doc = self.nlp(" ".join(sent))
        return [w.lemma for s in doc.sentences for w in s.words]


class BaseEvaluator(ABC):
    @abstractmethod
    def load_hypos_and_refs(self) -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        pass


class Evaluator(BaseEvaluator):
    METRIC_MAP = {
        "accuracy": Accuracy(),
        "recall": Recall(k=5),
        "distance": EditDistance(),
        "nlg": NLGMetrics()
    }

    def __init__(self, args: dict, metric_map: dict = None, no_lemma: bool = True):
        self.args = args
        self.metric_map = metric_map if metric_map else self.METRIC_MAP
        self.no_lemma = no_lemma
        self.nlp = StanfordNLPTool() if not no_lemma else None

    def load_hypos(self) -> List[List[List[str]]]:
        with open(self.args['RESULT_FILE'], 'r') as f:
            results = json.load(f)
        return self.load_hypos_raw(results)

    def load_hypos_raw(self, results) -> List[List[List[str]]]:
        # only use the first hypo
        assert type(results[0][0][0]) == list and type(results[0][0][1] == float), \
            "Each example should have a list of Hypothesis. Please prepare your result like " \
            "[Hypothesis(desc, score), ...]"
        # NOTE: results: List[List[list of tokens]]
        hypos = [[hypo[0] for hypo in r] for r in results]
        return hypos

    @staticmethod
    def normalize_hypos(hypos, src_references):
        new_hypos = []
        for hypo_list, src_sent in zip(hypos, src_references):
            if not hypo_list:
                print("find empty hypo list")
                hypo_list = [src_sent]
            new_hypos.append(hypo_list)
        return new_hypos

    def load_hypos_and_refs(self) -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        test_set = Dataset.create_from_file(self.args['TEST_SET'])
        references = list(test_set.get_ground_truth())
        src_references = list(test_set.get_src_descs())
        hypos = self.load_hypos()
        hypos = self.normalize_hypos(hypos, src_references)

        return hypos, references, src_references

    def _load_lemmas(self, origin: List, file_path: str, try_load: bool) -> List:
        if try_load and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        nlp = self.nlp
        if isinstance(origin[0][0], str):
            lemmas = [nlp.lemmatize(sent) for sent in origin]
        elif isinstance(origin[0][0], Iterable):
            lemmas = [[nlp.lemmatize(s) if s else [""] for s in sents] for sents in origin]
        else:
            raise TypeError("origin[0][0] should be str or Iterable, but is {}".format(type(origin[0])))
        with open(file_path, 'w') as f:
            json.dump(lemmas, f)
        return lemmas

    def prepare(self, hypos: List[List[List[str]]], references: List[List[str]], src_references: List[List[str]]) \
            -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        file_path = self.args['RESULT_FILE'] + '.l.hypos'
        lemma_hypos = self._load_lemmas(hypos, file_path, False)
        file_path = self.args['TEST_SET'] + '.l.refs'
        lemma_refs = self._load_lemmas(references, file_path, False)
        file_path = self.args['TEST_SET'] + '.l.src_refs'
        lemma_src_refs = self._load_lemmas(src_references, file_path, False)

        return lemma_hypos, lemma_refs, lemma_src_refs

    def cal_metrics(self, metrics: Iterable[str], hypos: List[List[List[str]]], references: List[List[str]],
                    src_references: List[List[str]]):
        results = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            results[metric] = instance.eval(hypos, references, src_references)
        return results

    def evaluate(self):
        metrics = self.args['--metrics'].split(',')
        hypos, references, src_references = self.load_hypos_and_refs()
        assert type(hypos[0][0]) == type(references[0])
        results = self.cal_metrics(metrics, hypos, references, src_references)
        logging.info(results)
        print(results)
        lemma_results = {}
        if not self.no_lemma:
            lemma_hypos, lemma_refs, lemma_src_refs = self.prepare(hypos, references, src_references)
            lemma_results = self.cal_metrics(metrics, lemma_hypos, lemma_refs, lemma_src_refs)
            logging.info(lemma_results)
            print(results)
        return results, lemma_results


class FracoEvaluator(Evaluator):
    def __init__(self, args: dict, metric_map: dict = None, no_lemma: bool = True):
        super(FracoEvaluator, self).__init__(args, metric_map, no_lemma)
        self.matched_count = 0

    @staticmethod
    def prepare_fraco_result_sent(r: dict) -> List[str]:
        return Tokenizer.tokenize_desc_with_con(r['result'])

    def load_hypos(self) -> List[List[List[str]]]:
        with open(self.args['RESULT_FILE'], 'r') as f:
            results = json.load(f)
        hypos = []
        for r in results:
            if r['matched']:
                self.matched_count += 1
            sent = self.prepare_fraco_result_sent(r)
            hypos.append([sent])
        return hypos

    def evaluate_with_raw_desc(self):
        match_count = 0
        correct_count = 0

        with open(self.args['RESULT_FILE'], 'r') as f:
            results = json.load(f)
        with open(self.args['TEST_SET'], 'r') as f:
            lines = f.readlines()
        assert len(results) == len(lines)
        for line, r in zip(lines, results):
            example = json.loads(line)
            if r['matched']:
                match_count += 1
                if r['result'].strip() == example['dst_desc'].strip():
                    correct_count += 1
            else:
                assert r['result'] == example['src_desc']
        return correct_count, match_count


def evaluate(args, no_lemma=True):
    EvalClass = globals()[args['--eval-class']]
    evaluator = EvalClass(args, no_lemma=no_lemma)
    return evaluator.evaluate()


def main():
    args = docopt(__doc__)
    evaluate(args, True)


if __name__ == '__main__':
    main()
