# encoding=utf-8
import json
import os

import fire

from utils.common import recover_desc
from eval import FracoEvaluator, NLGMetrics


def convert_result_to_readable(result_file="./CoAttnBPBAUpdater/result.json",
                               bleu_file="../results/CUP.bleu",
                               readable_file="../results/CUP.readable.txt",
                               is_fraco=False):
    print(result_file, bleu_file, readable_file)
    with open(result_file, 'r') as f:
        results = json.load(f)
    if is_fraco:
        sentences = [FracoEvaluator.prepare_fraco_result_sent(r) for r in results]
    else:
        sentences = [r[0][0] for r in results]
    with open(bleu_file, 'w') as f:
        for sent in sentences:
            f.write(NLGMetrics.prepare_sent(sent))
            f.write("\n")
        f.flush()
    with open(readable_file, 'w') as f:
        for sent in sentences:
            f.write(recover_desc(sent))
            f.write("\n")
        f.flush()


def _get_bleu_file(dir, prefix):
    return os.path.join(dir, prefix + ".bleu")


def _get_readable_file(dir, prefix):
    return os.path.join(dir, prefix + ".readable.txt")


def dump_all_readables(base_dir="./",
                       data_dir="../dataset/",
                       result_dir="../results/"):
    # create directories
    os.makedirs(result_dir, exist_ok=True)

    # dump CUP result
    print("Dump CUP result")
    convert_result_to_readable(result_file=os.path.join(base_dir, "CoAttnBPBAUpdater/result.json"),
                               bleu_file=_get_bleu_file(result_dir, "CUP"),
                               readable_file=_get_readable_file(result_dir, "CUP"))

    # dump reference result
    print("Extract Reference")
    ref_result = []
    with open(os.path.join(data_dir, "test.jsonl"), 'r') as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            # r[0][0]
            ref_result.append([[sample['dst_desc_tokens'], 1]])
    assert(len(ref_result) == 9673)
    with open(os.path.join(data_dir, "ref_result.json"), 'w') as f:
        json.dump(ref_result, f)
        f.flush()

    # dump baseline result
    print("Dump baseline results")
    baselines = {
        "nnupdater": ["nnupdater_result.json", "NNUpdater"],
        "fraco": ["fraco_result.json", "FracoUpdater"],
        "origin": ["origin_result.json", "Origin"],
        "ref": ["ref_result.json", "Reference"]

    }
    for base, values in baselines.items():
        convert_result_to_readable(result_file=os.path.join(data_dir, values[0]),
                                   bleu_file=_get_bleu_file(result_dir, values[1]),
                                   readable_file=_get_readable_file(result_dir, values[1]),
                                   is_fraco=True if base == 'fraco' else False)


if __name__ == '__main__':
    fire.Fire({
        "dump_all_readables": dump_all_readables
    })