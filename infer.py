# encoding=utf-8

"""
Infer procedure

Usage:
    infer.py [options] MODEL_PATH TEST_SET_FILE OUTPUT_FILE

Options:
    -h --help                   show this screen.
    --cuda                      use GPU
    --model-class=<str>         model class [default: models.updater.CoAttnBPBAUpdater]
    --seed=<int>                random seed [default: 0]
    --beam-size=<int>           beam size [default: 5]
    --max-dec-step=<int>        max decode steps [default: 100]
    --beam-class=<str>          beam class used for beam search [default: models.beam.Beam]
"""
from typing import Callable

import torch
import json
import logging
from tqdm import tqdm
from docopt import docopt
from dataset import Dataset, Example
from models.beam import Hypothesis
from utils.common import get_attr_by_name, set_reproducibility
from train import Procedure, List

logging.basicConfig(level=logging.INFO)


class Infer(Procedure):
    def __init__(self, args: dict):
        super(Infer, self).__init__(args)
        
    @property
    def beam_size(self):
        return int(self._args['--beam-size'])

    @property
    def max_dec_step(self):
        return int(self._args['--max-dec-step'])

    def _init_model(self):
        model_class = get_attr_by_name(self._args['--model-class'])
        self._model = model_class.load(self._args['MODEL_PATH'])
        self._set_device()

    def beam_search(self, data_set):
        logging.info("Using beam class: " + self._args['--beam-class'])
        BeamClass = get_attr_by_name(self._args['--beam-class'])
        was_training = self._model.training
        self._model.eval()

        hypos = []
        with torch.no_grad():
            for example in tqdm(data_set):
                example_hypos = self._model.beam_search(example, self.beam_size, self.max_dec_step, BeamClass)
                hypos.append(example_hypos)

        if was_training:
            self._model.train()
        return hypos

    def infer(self) -> List[List[Hypothesis]]:
        test_set = Dataset.create_from_file(self._args['TEST_SET_FILE'])
        self._init_model()
        hypos = self.beam_search(test_set)
        with open(self._args['OUTPUT_FILE'], 'w') as f:
            json.dump(hypos, f)
        return hypos

    def infer_one(self, code_change_seq: List[List[str]], src_desc_tokens: List[str],
                  ExampleClass: Callable = Example):
        self._init_model()
        example = ExampleClass.create_partial_example({
            'code_change_seq': code_change_seq,
            'src_desc_tokens': src_desc_tokens,
        })
        test_set = Dataset([example])
        self._init_model()
        hypos = self.beam_search(test_set)
        dst_desc_tokens = hypos[0][0][0]
        with open(self._args['OUTPUT_FILE'], 'w') as f:
            json.dump(hypos, f)
        return dst_desc_tokens


def infer(args):
    seed = int(args['--seed'])
    set_reproducibility(seed)

    infer_instance = Infer(args)
    infer_instance.infer()


def main():
    args = docopt(__doc__)
    infer(args)


if __name__ == '__main__':
    main()
