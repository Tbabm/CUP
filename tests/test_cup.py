# encoding=utf-8
import os
import unittest

from infer import Infer
from models.beam import Hypothesis
from train import Trainer, set_reproducibility

import logging

"""
# train logs for CUP:
epoch 1, iter 100, avg. loss 47.921413, avg. ppl 12.382730 cum. examples 3200, speed 2213.67 words/sec, time elapsed 27.53 sec
epoch 1, iter 200, avg. loss 30.874119, avg. ppl 4.850649 cum. examples 6400, speed 2268.12 words/sec, time elapsed 55.16 sec
epoch 1, iter 300, avg. loss 25.208198, avg. ppl 3.678934 cum. examples 9600, speed 2201.72 words/sec, time elapsed 83.30 sec
epoch 1, iter 400, avg. loss 22.864100, avg. ppl 3.265845 cum. examples 12800, speed 2215.05 words/sec, time elapsed 111.21 sec
epoch 1, iter 500, avg. loss 21.431996, avg. ppl 3.012126 cum. examples 16000, speed 2237.80 words/sec, time elapsed 139.01 sec
epoch 1, iter 500, cum. loss 29.659965, cum. ppl 4.634613 cum. examples 16000

# train logs for comment_updater
epoch 1, iter 100, avg. loss 47.921413, avg. ppl 12.382730 cum. examples 3200, speed 2274.32 words/sec, time elapsed 26.80 sec
epoch 1, iter 200, avg. loss 30.874119, avg. ppl 4.850649 cum. examples 6400, speed 2308.32 words/sec, time elapsed 53.91 sec
epoch 1, iter 300, avg. loss 25.208198, avg. ppl 3.678934 cum. examples 9600, speed 2261.88 words/sec, time elapsed 81.29 sec
epoch 1, iter 400, avg. loss 22.864100, avg. ppl 3.265845 cum. examples 12800, speed 2297.89 words/sec, time elapsed 108.20 sec
epoch 1, iter 500, avg. loss 21.431996, avg. ppl 3.012126 cum. examples 16000, speed 2298.36 words/sec, time elapsed 135.27 sec
epoch 1, iter 500, cum. loss 29.659965, cum. ppl 4.634613 cum. examples 16000
"""


class TestCUP(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_prefix = "tests/dataset"
        self.args = {
            '--cuda': True,
            '--train-data': os.path.join(self.dataset_prefix, 'train.jsonl'),
            '--dev-data': os.path.join(self.dataset_prefix, 'valid.jsonl'),
            '--vocab': os.path.join(self.dataset_prefix, 'mix_vocab.json'),
            '--vocab-embed': os.path.join(self.dataset_prefix, 'mix_vocab_embeddings.pkl'),
            '--model-class': "models.updater.CoAttnBPBAUpdater",
            '--embed-size': "300",
            '--edit-vec-size': "512",
            '--enc-hidden-size': "256",
            '--dec-hidden-size': "512",
            '--input-feed': True,
            '--share-embed': True,
            '--use-pre-embed': True,
            '--freeze-pre-embed': True,
            '--reload': False,
            '--mix-vocab': True,
            '--seed': "0",
            '--uniform-init': "0.1",
            '--train-batch-size': "8",
            '--valid-batch-size': "8",
            '--dropout': "0.2",
            '--teacher-forcing': "1.0",
            '--lr': "0.001",
            '--clip-grad': "5.0",
            '--log-every': "10",
            '--valid-niter': "500",
            '--patience': "5",
            '--max-trial-num': "5",
            '--lr-decay': "0.5",
            '--max-epoch': "2000",
            '--log-dir': "tests/output",
            '--save-to': "tests/output/model.bin",
            '--example-class': "dataset.Example",
            '--model-type': "generator"
        }

    def _test_train(self, trainer, train_set):
        losses = []
        for epoch in range(20):
            for batch in train_set.train_batch_iter(batch_size=trainer._train_batch_size, shuffle=False):
                batch_loss_val = trainer.train_a_batch(batch)
                losses.append(batch_loss_val)
        return losses

    def _test_valid(self, trainer, dev_set):
        valid_metric = trainer._validate(dev_set)
        return valid_metric

    def _test_infer(self):
        infer_args = {
            '--cuda': True,
            '--model-class': self.args['--model-class'],
            '--seed': "0",
            '--beam-size': "2",
            '--max-dec-step': "50",
            '--beam-class': "models.beam.Beam",
            '--model-type': "generator",
            '--batch-size': 3,
            'MODEL_PATH': os.path.join(self.args['--save-to']),
            'TEST_SET_FILE': os.path.join(self.dataset_prefix, "test.jsonl"),
            'OUTPUT_FILE': os.path.join(self.dataset_prefix, "result.json")
        }
        infer = Infer(infer_args)
        hypos = infer.infer()
        return hypos

    def test_reproduce(self):
        # disable all logging
        logging.disable(logging.CRITICAL)
        set_reproducibility(0)

        trainer = Trainer(self.args, tf_log=False)
        train_set, dev_set = trainer.load_dataset()
        trainer._init_model()

        expected_losses = [483.9374983910236, 396.7885422831847, 313.4529746738469, 412.6149972063738,
                           294.88370214788677, 276.9205821059003, 242.83269801561872, 225.63258253138852,
                           223.06413635684396, 209.42066629479157, 182.57185706984563, 199.2475491835321,
                           203.06883826238152, 166.84528868322133, 177.92200707610107, 163.95598015036012,
                           143.41510641020656, 164.14947102979863, 157.85085816150064, 145.01048472122704]
        losses = self._test_train(trainer, train_set)
        # print(losses)
        self.assertEqual(losses, expected_losses)

        expected_valid_loss = -19.518319110144216
        valid_loss = self._test_valid(trainer, dev_set)
        trainer.save_model()
        # print(valid_loss)
        self.assertEqual(valid_loss, expected_valid_loss)

        expected_hypos = [
            [Hypothesis(
                value=['Examines', 'the', 'undistorted', 'gray', 'input', 'image', 'for', 'squares', '<con>', '<con>',
                       '.', '.', '.'],
                score=-9.169617064004049),
             Hypothesis(
                 value=['Examines', 'the', 'undistorted', 'gray', 'input', 'image', 'for', 'squares', '<con>', '<con>',
                        '.', '.'],
                 score=-8.4981576175081)],
            [Hypothesis(
                value=['Save', 'basic', 'clusters', '<con>', '.', '.', '.', '.', '.'],
                score=-4.295478831250667),
             Hypothesis(
                 value=['Save', 'basic', 'clusters', '<con>', '.', '.', '.', '.', '.', '.'],
                 score=-4.809735736699733)],
            [Hypothesis(
                value=['Configure', 'a', 'ssl', '<con>', 'Config', 'for', 'the', 'server', 'using', 'the', 'legacy',
                       'legacy', 'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration', 'configuration', 'configuration',
                       'configuration', 'configuration', 'configuration'],
                score=-35.71404207185598)]
        ]
        hypos = self._test_infer()
        self.assertEqual(hypos, expected_hypos)
        # print(hypos)


if __name__ == '__main__':
    unittest.main()
