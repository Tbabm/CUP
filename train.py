# encoding=utf-8

"""
Training procedure

Usage:
    train.py --train-data=<file> --dev-data=<file> --vocab=<file> [options]

Options:
    -h --help                   show this screen.
    --cuda                      use GPU
    --train-data=<file>         training set file
    --dev-data=<file>           development set file
    --vocab=<file>              vocab file
    --model-class=<str>         model class [default: models.updater.CoAttnBPBAUpdater]
    --embed-size=<int>          embed size [default: 300]
    --edit-vec-size=<int>       edit vector size [default: 512]
    --enc-hidden-size=<int>     encoder hidden size [default: 256]
    --dec-hidden-size=<int>     hidden size [default: 512]
    --input-feed                use input feeding
    --share-embed               share the embeddings of src_encoder and editor
    --mix-vocab                 mix the vocabs of code and nl
    --seed=<int>                random seed [default: 0]
    --use-pre-embed             use pre-trained embeddings to initialize word embeddings
    --freeze-pre-embed          freeze the pre-trained embeddings
    --vocab-embed=<file>        the pre-built vocab embeddings [default: vocab_embeddings.pkl]
    --uniform-init=<float>      uniform initialization of parameters [default: 0.1]
    --train-batch-size=<int>    train batch size [default: 32]
    --valid-batch-size=<int>    valid batch size [default: 32]
    --lr=<float>                learning rate [default: 0.001]
    --dropout=<float>           dropout rate [default: 0.0]
    --teacher-forcing=<float>   teacher forcing ratio [default: 1.0]
    --clip-grad=<float>         gradient clipping [default: 5.0]
    --log-every=<int>           log interval [default: 100]
    --valid-niter=<int>         validate interval [default: 500]
    --patience=<int>            wait for how many validations to decay learning rate [default: 5]
    --max-trial-num=<int>       terminal training after how many trials [default: 5]
    --lr-decay=<float>          learning rate decay [default: 0.5]
    --max-epoch=<int>           max epoch [default: 50]
    --log-dir=<dir>             dir for tensorboard log [default: log/]
    --save-to=<file>            model save path [default: model.bin]
    --example-class=<str>       Example Class used to load an example [default: dataset.Example]
"""

# Reference: https://github.com/pcyin/pytorch_basic_nmt

import time
from abc import ABC, abstractmethod
from docopt import docopt
import logging
from utils.common import *
import tensorflow as tf
from dataset import Dataset, Batch


class TFLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def scalar_dict_summary(self, info, step):
        for tag, value in info.items():
            self.scalar_summary(tag, value, step)


class LossReporter(object):
    def __init__(self, tf_logger: TFLogger = None):
        self._report_loss = 0
        self._cum_loss = 0
        self._report_tgt_words = 0
        self._cum_tgt_words = 0
        self._report_examples = 0
        self._cum_examples = 0
        self._train_begin_time = self._begin_time = time.time()
        self.tf_logger = tf_logger

    @property
    def report_tgt_words(self):
        return self._report_tgt_words

    @property
    def avg_loss_per_example(self):
        return self._report_loss / self._report_examples

    @property
    def avg_ppl(self):
        return np.exp(self._report_loss / self._report_tgt_words)

    @property
    def avg_cum_loss_per_example(self):
        return self._cum_loss / self._cum_examples

    @property
    def avg_cum_ppl(self):
        return np.exp(self._cum_loss / self._cum_tgt_words)

    def update(self, batch_loss, tgt_words_num, batch_size):
        self._report_loss += batch_loss
        self._cum_loss += batch_loss
        self._report_tgt_words += tgt_words_num
        self._cum_tgt_words += tgt_words_num
        self._report_examples += batch_size
        self._cum_examples += batch_size

    def reset_report_stat(self):
        self._report_loss = 0
        self._report_tgt_words = 0
        self._report_examples = 0
        self._train_begin_time = time.time()

    def reset_cum_stat(self):
        self._cum_loss = 0
        self._cum_examples = 0
        self._cum_tgt_words = 0

    def report(self, epoch, iter):
        train_time = time.time() - self._train_begin_time
        spend_time = time.time() - self._begin_time
        logging.info('epoch %d, iter %d, avg. loss %.6f, avg. ppl %.6f ' \
                     'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec'
                     % (epoch, iter, self.avg_loss_per_example, self.avg_ppl,
                        self._cum_examples, self.report_tgt_words / train_time, spend_time))
        if self.tf_logger:
            tf_info = {
                'train_loss': self.avg_loss_per_example,
                'train_ppl': self.avg_ppl,
            }
            self.tf_logger.scalar_dict_summary(tf_info, iter)

    def report_cum(self, epoch, iter):
        logging.info('epoch %d, iter %d, cum. loss %.6f, cum. ppl %.6f cum. examples %d'
                     % (epoch, iter, self.avg_cum_loss_per_example, self.avg_cum_ppl, self._cum_examples))
        if self.tf_logger:
            tf_info = {
                'cum_loss': self.avg_cum_loss_per_example,
                'cum_ppl': self.avg_cum_ppl
            }
            self.tf_logger.scalar_dict_summary(tf_info, iter)

    def report_valid(self, iter, ppl):
        logging.info('validation: iter %d, dev. ppl %f' % (iter, ppl))
        if self.tf_logger:
            self.tf_logger.scalar_summary("ppl", ppl, iter)


class Procedure(ABC):
    def __init__(self, args: dict):
        self._args = args
        self._model = None

    def _set_device(self):
        self._device = torch.device("cuda:0" if self._args['--cuda'] else "cpu")
        logging.info("use device: {}".format(self._device))
        self._model.to(self._device)

    @abstractmethod
    def _init_model(self):
        pass


class Trainer(Procedure):
    def __init__(self, args: dict, tf_log: bool = True):
        super(Trainer, self).__init__(args)
        self._device = None
        self._cur_patience = 0
        self._cur_trail = 0
        self._hist_valid_scores = []
        self.tf_logger = TFLogger(self._args['--log-dir']) if tf_log else None

    @property
    def _train_batch_size(self):
        return int(self._args['--train-batch-size'])

    @property
    def _valid_batch_size(self):
        return int(self._args['--valid-batch-size'])

    @property
    def _clip_grad(self):
        return float(self._args['--clip-grad'])

    @property
    def _log_every(self):
        return int(self._args['--log-every'])

    @property
    def _valid_niter(self):
        return int(self._args['--valid-niter'])

    @property
    def _model_save_path(self):
        return self._args['--save-to']

    @property
    def _max_patience(self):
        return int(self._args['--patience'])

    @property
    def _max_trial_num(self):
        return int(self._args['--max-trial-num'])

    @property
    def _max_epoch(self):
        return int(self._args['--max-epoch'])

    @property
    def _optim_save_path(self):
        return self._model_save_path + '.optim'

    def _uniform_init_model_params(self):
        uniform_init = float(self._args['--uniform-init'])
        if np.abs(uniform_init) > 0.:
            logging.info('uniformly initialize parameters [-{}, +{}]'.format(uniform_init, uniform_init))
            for p in self._model.parameters():
                p.data.uniform_(-uniform_init, uniform_init)

    def _init_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=float(self._args['--lr']))

    def train_a_batch(self, batch: Batch) -> float:
        self._optimizer.zero_grad()
        # (batch_size)
        example_losses = self._model(batch)
        batch_loss = example_losses.sum()
        loss = batch_loss / len(batch)
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()
        return batch_loss.item()

    def save_model(self):
        logging.info('save currently the best model to [%s]' % self._model_save_path)
        self._model.save(self._model_save_path, self._args)
        # also save the optimizers' state
        torch.save(self._optimizer.state_dict(), self._optim_save_path)

    def load_model(self):
        logging.info('load previously best model')
        params = torch.load(self._model_save_path, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(params['state_dict'])
        self._model.to(self._device)

        logging.info('restore parameters of the optimizers')
        self._optimizer.load_state_dict(torch.load(self._optim_save_path))

    def decay_lr(self):
        # decay lr, and restore from previously best checkpoint
        lr = self._optimizer.param_groups[0]['lr'] * float(self._args['--lr-decay'])
        logging.info('decay learning rate to %f' % lr)
        self.load_model()

        # set new lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _validate(self, dev_set):
        was_training = self._model.training
        self._model.eval()

        cum_loss = 0
        cum_tgt_words = 0
        with torch.no_grad():
            for batch in dev_set.train_batch_iter(self._valid_batch_size, shuffle=False):
                batch_loss = self._model(batch).sum()
                cum_loss += batch_loss.item()
                cum_tgt_words += batch.tgt_words_num
            dev_ppl = np.exp(cum_loss / cum_tgt_words)
        # negative: the larger the better
        valid_metric = -dev_ppl

        if was_training:
            self._model.train()

        return valid_metric

    def validate(self, train_iter, dev_set, loss_reporter):
        logging.info('begin validation ...')

        valid_metric = self._validate(dev_set)
        loss_reporter.report_valid(train_iter, valid_metric)

        is_better = len(self._hist_valid_scores) == 0 or valid_metric > max(self._hist_valid_scores)
        self._hist_valid_scores.append(valid_metric)

        return is_better

    def _init_model(self):
        model_class = get_attr_by_name(self._args['--model-class'])
        self._model = model_class(*model_class.prepare_model_params(self._args))
        self._model.train()

        self._uniform_init_model_params()

        freeze = bool(self._args['--freeze-pre-embed'])
        if bool(self._args['--use-pre-embed']):
            logging.info("initialize word embeddings with pretrained embeddings")
            self._model.vocab.load_embeddings(self._args['--vocab-embed'])
            self._model.init_pretrain_embeddings(freeze)

        self._set_device()
        self._init_optimizer()

    def load_dataset(self):
        logging.info("Load example using {}".format(self._args['--example-class']))
        example_class = get_attr_by_name(self._args['--example-class'])
        train_set = Dataset.create_from_file(self._args['--train-data'], example_class)
        dev_set = Dataset.create_from_file(self._args['--dev-data'], example_class)
        return train_set, dev_set

    def train(self):
        train_set, dev_set = self.load_dataset()
        self._init_model()

        epoch = train_iter = 0
        loss_reporter = LossReporter(self.tf_logger)
        logging.info("Start training")
        while True:
            epoch += 1
            for batch in train_set.train_batch_iter(batch_size=self._train_batch_size, shuffle=True):
                train_iter += 1
                batch_loss_val = self.train_a_batch(batch)
                # tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                loss_reporter.update(batch_loss_val, batch.tgt_words_num, len(batch))

                if train_iter % self._log_every == 0:
                    loss_reporter.report(epoch, train_iter)
                    loss_reporter.reset_report_stat()

                if train_iter % self._valid_niter == 0:
                    loss_reporter.report_cum(epoch, train_iter)
                    loss_reporter.reset_cum_stat()

                    is_better = self.validate(train_iter, dev_set, loss_reporter)
                    if is_better:
                        self._cur_patience = 0
                        self.save_model()
                    else:
                        self._cur_patience += 1
                        logging.info('hit patience {}'.format(self._cur_patience))

                        if self._cur_patience == self._max_patience:
                            self._cur_trail += 1
                            logging.info('hit #{} trial'.format(self._cur_trail))
                            if self._cur_trail == self._max_trial_num:
                                logging.info('early stop!')
                                return

                            self.decay_lr()

                            # reset patience
                            self._cur_patience = 0

            if epoch == self._max_epoch:
                logging.info('reached maximum number of epochs')
                return


def train(args):
    logging.debug("Train with args:")
    logging.info(args)

    seed = int(args['--seed'])
    set_reproducibility(seed)

    trainer = Trainer(args)
    trainer.train()


def main():
    args = docopt(__doc__)
    train(args)


if __name__ == '__main__':
    main()
