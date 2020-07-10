# encoding=utf-8

import os
from collections import OrderedDict

import fire
import json
import numpy as np
import subprocess

from .nnupdater import slice_test_main
from eval import Evaluator, FracoEvaluator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def run_origin(test_file="../dataset/test.jsonl",
               out_file="../dataset/origin_result.json"):
    with open(test_file, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        sample = json.loads(line)
        results.append([(sample['src_desc_tokens'], 1)])
    with open(out_file, 'w') as f:
        json.dump(results, f)


def run_fraco(test_set="../dataset/test.jsonl",
              out_file="../dataset/fraco_result.json",
              fraco_dir="../FracoUpdater/"):
    test_set = os.path.abspath(test_set)
    out_file = os.path.abspath(out_file)
    cmd = 'cd {} && mvn compile && mvn exec:java -Dexec.args="{} {}"'.format(fraco_dir, test_set, out_file)
    subprocess.run(cmd, shell=True, check=True)


def tune_nnupdater(train_file="../dataset/train.jsonl",
                   valid_file="../dataset/valid.jsonl",
                   dump_file="../dataset/nnupdater_tune_results.json",
                   slice_count=4):
    # get best vectorizer with alpha=0.5
    diff_vec_classes = [CountVectorizer, TfidfVectorizer]
    desc_vec_classes = [CountVectorizer, TfidfVectorizer]

    def _get_class_name(clazz):
        return str(clazz).split('.')[-1][:-3]

    def _train_eval(diff_vec_class, desc_vec_class, alp):
        metrics = "accuracy,recall,distance,nlg"
        suf = "{}_{}_{}".format(_get_class_name(diff_vec_class), _get_class_name(desc_vec_class), alp)
        out_file = os.path.join(os.path.dirname(valid_file),
                                "nnupdater_result_{}.json".format(suf))
        slice_test_main(train_file, valid_file, out_file, alpha=alp, slice_count=slice_count,
                        diff_vec_class=diff_vec_class, desc_vec_class=desc_vec_class)
        evaluator = Evaluator(args={
            "--metrics": metrics,
            "TEST_SET": valid_file,
            "RESULT_FILE": out_file
        })
        r, lemma_r = evaluator.evaluate()
        return suf, r, lemma_r

    result_dict = OrderedDict()
    for diff_vec in diff_vec_classes:
        for desc_vec in desc_vec_classes:
            suffix, result, lemma_result = _train_eval(diff_vec, desc_vec, 0.5)
            result_dict[suffix] = {
                'diff_vec': diff_vec,
                'desc_vec': desc_vec
            }
            result_dict[suffix]['result'] = result
    print(result_dict)
    items = sorted(list(result_dict.items()),
                   key=lambda x: (x[1]['result']['accuracy']['accuracy'], x[1]['result']['nlg']['hypo']['Bleu_4']),
                   reverse=True)
    best_item = items[0]
    best_diff_vec = best_item[1]['diff_vec']
    best_desc_vec = best_item[1]['desc_vec']
    print("best diff vec: {}".format(best_diff_vec))
    print("best desc vec: {}".format(best_desc_vec))

    for alpha in np.arange(0, 1.1, 0.1):
        suffix, result, lemma_result = _train_eval(best_diff_vec, best_desc_vec, alpha)
        result_dict[suffix] = {
            'result': result
        }
    items = sorted(list(result_dict.items()),
                   key=lambda x: (x[1]['result']['accuracy']['accuracy'], x[1]['result']['nlg']['hypo']['Bleu_4']),
                   reverse=True)
    print(items[0])

    # deal with type object
    class TypeEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, type):
                return str(o)
            return super().default(self, o)

    with open(dump_file, 'w') as f:
        json.dump(items, f, cls=TypeEncoder)


def run_nnupdater(train_file="../dataset/train.jsonl",
                  test_file="../dataset/test.jsonl",
                  out_file="../dataset/nnupdater_result.json",
                  # the default value is the best after tuning
                  alpha=0,
                  slice_count=4):
    slice_test_main(train_file, test_file, out_file, alpha, slice_count, diff_vec_class=TfidfVectorizer,
                    desc_vec_class=TfidfVectorizer)


def run_all_baselines(train_set="../dataset/train.jsonl",
                      test_set="../dataset/test.jsonl",
                      out_prefix="../dataset/",
                      fraco_dir="../FracoUpdater/"):
    print("run Origin")
    origin_out = os.path.join(out_prefix, "origin_result.json")
    run_origin(test_set, origin_out)

    print("run Fraco")
    fraco_out = os.path.join(out_prefix, "fraco_result.json")
    run_fraco(test_set, fraco_out, fraco_dir)

    print("run nnupdater")
    nn_out = os.path.join(out_prefix, "nnupdater_result.json")
    run_nnupdater(train_set, test_set, nn_out)

    print("Done!")


def eval_one_app(app: str, value_dict: dict, test_set: str) -> dict:
    metrics = "accuracy,recall,distance,nlg"
    result_file = value_dict['result_file']
    print("eval {}".format(app))
    if app == "fraco":
        EvalClass = FracoEvaluator
    else:
        EvalClass = Evaluator
    evaluator = EvalClass(args={
        "--metrics": metrics,
        "TEST_SET": test_set,
        "RESULT_FILE": result_file
    })
    r, lemma_r = evaluator.evaluate()
    value_dict['result'] = r
    return value_dict


def eval_approaches(approaches: dict, test_set: str) -> dict:
    for app, value in approaches.items():
        eval_one_app(app, value, test_set)
    return approaches


def collect_results(test_set="../dataset/test.jsonl",
                    origin_result="../dataset/origin_result.json",
                    fraco_result="../dataset/fraco_result.json",
                    nnupdater_result="../dataset/nnupdater_result.json",
                    my_result="CoAttnBPBAUpdater/result.json",
                    out_file="../dataset/all_result.json"):
    approaches = OrderedDict({
        "origin": {
            "result_file": origin_result,
        },
        "fraco": {
            "result_file": fraco_result,
        },
        "nnupdater": {
            "result_file": nnupdater_result,
        },
        "my": {
            "result_file": my_result,
        },
    })
    approaches = eval_approaches(approaches, test_set)
    with open(out_file, 'w') as f:
        json.dump(approaches, f, indent=2)

    return approaches


if __name__ == '__main__':
    fire.Fire({
        "run_origin": run_origin,
        "run_fraco": run_fraco,
        "tune_nnupdater": tune_nnupdater,
        "run_nnupdater": run_nnupdater,
        "run_all_baselines": run_all_baselines,
        "collect_results": collect_results
    })
