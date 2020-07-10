# encoding=utf-8

"""
Comment Updater Model
"""

import logging

import models.base
import models.encoder
from models import base
from models.beam import Beam
from dataset import Batch
from vocab import Vocab, MixVocab
from .decoder import *
from .encoder import SeqEditEncoder, CoAttnLayer, ModelingLayer
from .utils import negative_log_likelihood, dot_prod_attention

logging.basicConfig(level=logging.INFO)


class BaseUpdater(base.BaseModel, ABC):
    @staticmethod
    def log_args(**kwargs):
        logging.info("Create model using parameters:")
        for key, value in kwargs.items():
            logging.info("{}={}".format(key, value))

    @staticmethod
    def prepare_model_params(args):
        vocab_class = MixVocab if args.get('--mix-vocab', False) else Vocab
        return int(args['--embed-size']), int(args['--edit-vec-size']), int(args['--enc-hidden-size']), \
               int(args['--dec-hidden-size']), vocab_class.load(args['--vocab']), args

    def __init__(self):
        super(BaseUpdater, self).__init__()
        self.mix_vocab = None
        self.share_embed = None

        self.vocab = None
        self.code_vocab = None
        self.nl_vocab = None
        self.action_vocab = None

        self.code_embed_layer = None
        self.action_embed_layer = None
        self.enc_nl_embed_layer = None
        self.dec_nl_embed_layer = None

        self.code_edit_encoder = None
        self.nl_encoder = None
        self.dec_state_init = None
        self.nl_decoder = None

    def init_embeddings(self, args, vocab: Union[Vocab, MixVocab], embed_size):
        self.mix_vocab = bool(args.get('--mix-vocab', False))
        self.share_embed = bool(args['--share-embed'])

        self.vocab = vocab
        self.code_vocab = vocab.code if not self.mix_vocab else vocab.token
        self.nl_vocab = vocab.nl if not self.mix_vocab else vocab.token
        self.action_vocab = vocab.action

        self.enc_nl_embed_layer = nn.Embedding(len(self.nl_vocab), embed_size, padding_idx=self.nl_vocab[PADDING])
        self.action_embed_layer = nn.Embedding(len(self.action_vocab), embed_size,
                                               padding_idx=self.action_vocab[PADDING])
        if self.share_embed:
            logging.info("Encoder and decoder share embedings")
            self.dec_nl_embed_layer = self.enc_nl_embed_layer
        else:
            self.dec_nl_embed_layer = nn.Embedding(len(self.nl_vocab), embed_size, padding_idx=self.nl_vocab[PADDING])

        if self.mix_vocab:
            logging.info("Code and nl share embeddings")
            self.code_embed_layer = self.enc_nl_embed_layer
        else:
            self.code_embed_layer = nn.Embedding(len(self.code_vocab), embed_size,
                                                 padding_idx=self.code_vocab[PADDING])

    def init_pretrain_embeddings(self, freeze: bool):
        self.enc_nl_embed_layer.weight.data.copy_(torch.from_numpy(self.nl_vocab.embeddings))
        self.enc_nl_embed_layer.weight.requires_grad = not freeze
        if not self.share_embed:
            self.dec_nl_embed_layer.weight.data.copy_(torch.from_numpy(self.nl_vocab.embeddings))
            self.dec_nl_embed_layer.weight.requires_grad = not freeze
        if not self.mix_vocab:
            self.code_embed_layer.weight.data.copy_(torch.from_numpy(self.code_vocab.embeddings))
            self.code_embed_layer.weight.requires_grad = not freeze

    @property
    def device(self) -> torch.device:
        return self.nl_encoder.embed_layer.weight.device

    def _prepare_dec_init_state(self, last_cells: List[Tensor]):
        """
        :param last_cells: List[(batch_size, edit_vec_size)]
        :return:
            (batch_size, hidden_size)
            (batch_size, hidden_size)
        """
        # (batch_size, hidden_size)
        dec_init_cell = self.dec_state_init(torch.cat(last_cells, dim=-1))
        dec_init_state = torch.tanh(dec_init_cell)
        return dec_init_state, dec_init_cell

    def _get_sent_masks(self, max_len: int, sent_lens: List[int]):
        src_sent_masks = torch.zeros(len(sent_lens), max_len, dtype=FLOAT_TYPE)
        for e_id, l in enumerate(sent_lens):
            # make all paddings to 1
            src_sent_masks[e_id, l:] = 1
        return src_sent_masks.to(self.device)

    def construct_src_input(self, batch):
        code_tensor_a, code_tensor_b, action_tensor = batch.get_code_change_tensors(self.code_vocab, self.action_vocab,
                                                                                    self.device)
        src_tensor = batch.get_src_tensor(self.nl_vocab, self.device)
        code_lens = batch.get_code_lens()
        src_lens = batch.get_src_lens()
        return code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens

    @property
    def encoder_output_names(self):
        return ["edit_encodings", "edit_last_state", "edit_sent_masks", "src_encodings", "src_sent_masks",
                "dec_init_state"]

    def encode(self, code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens, *args) -> dict:
        # encodings: (batch_size, sent_len, hidden_size * direction * #layer)
        edit_encodings, edit_last_state, edit_last_cell = self.code_edit_encoder(code_tensor_a, code_tensor_b,
                                                                                 action_tensor, code_lens)
        src_encodings, src_last_state, src_last_cell = self.nl_encoder(src_tensor, src_lens)
        dec_init_state = self._prepare_dec_init_state([src_last_cell, edit_last_cell])
        src_sent_masks = self._get_sent_masks(src_encodings.size(1), src_lens)
        edit_sent_masks = self._get_sent_masks(edit_encodings.size(1), code_lens)
        local_vars = locals().copy()
        encoder_output = {name: local_vars[name] for name in self.encoder_output_names}
        return encoder_output

    @abstractmethod
    def prepare_tgt_out_tensor(self, batch: Batch) -> Tensor:
        return batch.get_tgt_out_tensor(self.nl_vocab, self.device)

    @abstractmethod
    def prepare_decoder_kwargs(self, encoder_output: dict, batch: Batch) -> dict:
        pass

    def forward(self, batch: Batch) -> Tensor:
        input_tensor = self.construct_src_input(batch)
        tgt_in_tensor = batch.get_tgt_in_tensor(self.nl_vocab, self.device)
        tgt_out_tensor = self.prepare_tgt_out_tensor(batch)
        encoder_output = self.encode(*input_tensor)

        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        # omit the last word of tgt, which is </s>
        # (tgt_sent_len - 1, batch_size, hidden_size)
        word_losses, ys = self.nl_decoder(tgt_in_tensor, tgt_out_tensor, **decoder_kwargs)

        # (batch_size,)
        example_losses = word_losses.sum(dim=0)
        return example_losses

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int,
                    BeamClass=Beam) -> List[Hypothesis]:
        batch = Batch([example])
        input_tensor = self.construct_src_input(batch)
        encoder_output = self.encode(*input_tensor)
        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        hypos = self.nl_decoder.beam_search(example, beam_size, max_dec_step, BeamClass, **decoder_kwargs)
        return hypos


class BaseBothPtrUpdater(BaseUpdater, ABC):
    def prepare_tgt_out_tensor(self, batch: Batch) -> Tensor:
        return batch.get_both_ext_tgt_out_tensor(self.nl_vocab, self.device)

    def prepare_decoder_kwargs(self, encoder_output: dict, batch: Batch) -> dict:
        names = ['src_encodings', 'edit_encodings', 'src_sent_masks', 'edit_sent_masks', 'dec_init_state']
        kwargs = {name: encoder_output[name] for name in names}
        kwargs['src_ext_tensor'], kwargs['code_ext_tensor'] = batch.get_both_ext_tensor(self.nl_vocab, self.device)
        kwargs['max_ext_size'] = batch.get_max_both_ext_size()
        return kwargs


class BothPtrBAUpdater(BaseBothPtrUpdater):
    """
    Updater with two pointers and two attentions
    """

    def __init__(self, embed_size: int, edit_vec_size: int, enc_hidden_size: int, dec_hidden_size: int,
                 vocab: Union[Vocab, MixVocab], args: dict,
                 loss_func: Callable = negative_log_likelihood,
                 att_func: Callable = dot_prod_attention):
        super(BothPtrBAUpdater, self).__init__()
        params = locals().copy()
        params.pop('self')
        params.pop('__class__')
        self.log_args(**params)

        self.init_embeddings(args, vocab, embed_size)
        dropout_rate = float(args['--dropout'])

        # code edit encoder
        # (code_len, batch_size) -> [(code_len, batch_size, edit_vec_size), (batch_size, edit_vec_size)]
        self.code_edit_encoder = SeqEditEncoder(embed_size, edit_vec_size, self.code_embed_layer,
                                                self.action_embed_layer, dropout=dropout_rate)
        self.nl_encoder = models.encoder.Encoder(embed_size, enc_hidden_size, self.enc_nl_embed_layer,
                                                 dropout=dropout_rate)
        self.dec_state_init = nn.Linear(self.code_edit_encoder.output_size + self.nl_encoder.output_size,
                                        dec_hidden_size)
        self.nl_decoder = BothPtrBASeqEditor(self.code_edit_encoder.output_size, self.nl_encoder.output_size,
                                             embed_size, dec_hidden_size, self.nl_vocab, self.dec_nl_embed_layer,
                                             args, att_func, loss_func)


class CoAttnBPBAUpdater(BaseBothPtrUpdater):
    """
    Use co-attention mechanism in Encoder
    """

    def __init__(self, embed_size: int, edit_vec_size: int, enc_hidden_size: int, dec_hidden_size: int,
                 vocab: Union[Vocab, MixVocab], args: dict,
                 loss_func: Callable = negative_log_likelihood,
                 att_func: Callable = dot_prod_attention):
        super(CoAttnBPBAUpdater, self).__init__()
        params = locals().copy()
        params.pop('self')
        params.pop('__class__')
        self.log_args(**params)

        self.init_embeddings(args, vocab, embed_size)
        dropout_rate = float(args['--dropout'])

        # code edit encoder
        # (code_len, batch_size) -> [(code_len, batch_size, edit_vec_size), (batch_size, edit_vec_size)]
        self.code_edit_encoder = SeqEditEncoder(embed_size, edit_vec_size, self.code_embed_layer,
                                                self.action_embed_layer, dropout=dropout_rate)
        self.nl_encoder = models.encoder.Encoder(embed_size, enc_hidden_size, self.enc_nl_embed_layer,
                                                 dropout=dropout_rate)

        # for co-attn
        self.co_attn_layer = CoAttnLayer(self.code_edit_encoder.output_size, self.nl_encoder.output_size)
        self.code_edit_modeling_layer = ModelingLayer(self.code_edit_encoder.output_size + self.nl_encoder.output_size,
                                                      enc_hidden_size, dropout_rate)
        self.nl_modeling_layer = ModelingLayer(self.nl_encoder.output_size + self.code_edit_encoder.output_size,
                                               enc_hidden_size, dropout_rate)

        self.dec_state_init = nn.Linear(self.code_edit_encoder.output_size + self.nl_encoder.output_size,
                                        dec_hidden_size)
        self.nl_decoder = BothPtrBASeqEditor(self.code_edit_encoder.output_size, self.nl_encoder.output_size,
                                             embed_size, dec_hidden_size, self.nl_vocab, self.dec_nl_embed_layer,
                                             args, att_func, loss_func)

    def encode(self, code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens, *args) -> dict:
        # encodings: (batch_size, sent_len, hidden_size * direction * #layer)
        c_edit_encodings, c_edit_last_state, c_edit_last_cell = self.code_edit_encoder(code_tensor_a, code_tensor_b,
                                                                                       action_tensor, code_lens)
        c_src_encodings, c_src_last_state, c_src_last_cell = self.nl_encoder(src_tensor, src_lens)
        edit_sent_masks = self._get_sent_masks(c_edit_encodings.size(1), code_lens)
        src_sent_masks = self._get_sent_masks(c_src_encodings.size(1), src_lens)

        # for co-attention
        edit_ctx_encodings, src_ctx_encodings = self.co_attn_layer(c_edit_encodings, c_src_encodings, edit_sent_masks,
                                                                   src_sent_masks)
        edit_modeling_input = torch.cat([c_edit_encodings, edit_ctx_encodings], dim=-1)
        edit_encodings, edit_last_state, edit_last_cell = self.code_edit_modeling_layer(edit_modeling_input, code_lens)
        src_modeling_input = torch.cat([c_src_encodings, src_ctx_encodings], dim=-1)
        src_encodings, src_last_state, src_last_cell = self.nl_modeling_layer(src_modeling_input, src_lens)

        dec_init_state = self._prepare_dec_init_state([src_last_cell, edit_last_cell])
        local_vars = locals().copy()
        encoder_output = {name: local_vars[name] for name in self.encoder_output_names}
        return encoder_output


class NoCoAttnBPBAUpdater(BaseBothPtrUpdater):
    """
    Remove co-attention mechanism in Encoder
    """

    def __init__(self, embed_size: int, edit_vec_size: int, enc_hidden_size: int, dec_hidden_size: int,
                 vocab: Union[Vocab, MixVocab], args: dict,
                 loss_func: Callable = negative_log_likelihood,
                 att_func: Callable = dot_prod_attention):
        super(NoCoAttnBPBAUpdater, self).__init__()
        params = locals().copy()
        params.pop('self')
        params.pop('__class__')
        self.log_args(**params)

        self.init_embeddings(args, vocab, embed_size)
        dropout_rate = float(args['--dropout'])

        # code edit encoder
        # (code_len, batch_size) -> [(code_len, batch_size, edit_vec_size), (batch_size, edit_vec_size)]
        self.code_edit_encoder = SeqEditEncoder(embed_size, edit_vec_size, self.code_embed_layer,
                                                self.action_embed_layer, dropout=dropout_rate)
        self.nl_encoder = models.encoder.Encoder(embed_size, enc_hidden_size, self.enc_nl_embed_layer,
                                                 dropout=dropout_rate)

        # NOTE: only keep the modeling layer
        self.code_edit_modeling_layer = ModelingLayer(self.code_edit_encoder.output_size, enc_hidden_size, dropout_rate)
        self.nl_modeling_layer = ModelingLayer(self.nl_encoder.output_size, enc_hidden_size, dropout_rate)

        self.dec_state_init = nn.Linear(self.code_edit_encoder.output_size + self.nl_encoder.output_size,
                                        dec_hidden_size)
        self.nl_decoder = BothPtrBASeqEditor(self.code_edit_encoder.output_size, self.nl_encoder.output_size,
                                             embed_size, dec_hidden_size, self.nl_vocab, self.dec_nl_embed_layer,
                                             args, att_func, loss_func)

    def encode(self, code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens, *args) -> dict:
        # encodings: (batch_size, sent_len, hidden_size * direction * #layer)
        c_edit_encodings, c_edit_last_state, c_edit_last_cell = self.code_edit_encoder(code_tensor_a, code_tensor_b,
                                                                                       action_tensor, code_lens)
        c_src_encodings, c_src_last_state, c_src_last_cell = self.nl_encoder(src_tensor, src_lens)
        edit_sent_masks = self._get_sent_masks(c_edit_encodings.size(1), code_lens)
        src_sent_masks = self._get_sent_masks(c_src_encodings.size(1), src_lens)

        edit_encodings, edit_last_state, edit_last_cell = self.code_edit_modeling_layer(c_edit_encodings, code_lens)
        src_encodings, src_last_state, src_last_cell = self.nl_modeling_layer(c_src_encodings, src_lens)

        dec_init_state = self._prepare_dec_init_state([src_last_cell, edit_last_cell])
        local_vars = locals().copy()
        encoder_output = {name: local_vars[name] for name in self.encoder_output_names}
        return encoder_output
