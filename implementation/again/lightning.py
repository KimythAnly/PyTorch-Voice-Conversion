from collections import OrderedDict

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange, reduce

from .model import Encoder, Decoder, Activation
from ..classifier.model import Classifier
from ..util.lit import BaseModel
from ..util.torch import get_optim
from ..vocoder.melgan import Vocoder


class Model(BaseModel):
    def __init__(
        self,
        dataset,
        batch_size,
        encoder_params,
        decoder_params,
        activation_params,
        optimizer_config,
        classifier_params,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # Model
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.act = Activation(**activation_params)
        # Optimizer
        self.optimizer_config = optimizer_config
        # Vocoder
        self.vocoder = Vocoder(device=self.device)
        # Classifier
        self.classifier = Classifier(**classifier_params)
        # criterion
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop. It is independent of forward
        sid = batch['sid']
        mel = batch['mel']
        pos = batch['pos']

        should_log = (self.trainer.global_step +
                      1) % self.trainer.log_every_n_steps == 0

        if optimizer_idx == 0:
            rec = self._model(mel)
            loss = self.criterion_l1(rec, mel)
            if should_log:
                with torch.no_grad():
                    dec = self.forward(mel[[0]], mel[[1]])
                    log_mels = {
                        'src': mel[0],
                        'tgt': mel[1],
                        'dec': dec[0],
                        'rec': rec[0],
                    }
                    log_mels_plt = {
                        k: self.vocoder.mel2plt(v) for k, v in log_mels.items()
                    }
                    log_audios = {
                        k: self.vocoder.mel2wav(v) for k, v in log_mels.items()
                    }
                    log_scalars = {
                        'rec_loss': loss
                    }

                self.log_scalars(log_scalars, step=self.trainer.global_step)
                self.log_mels('train/mel', log_mels_plt,
                              step=self.trainer.global_step)
                self.log_audios('train/audio', log_audios,
                                step=self.trainer.global_step)
            log_bar_dict = {
                'rec_loss': round(loss.item(), 3),
            }

        if optimizer_idx == 1:
            c, _, _ = self.encoder(mel)
            c = self.act(c)
            pred = self.classifier(c.detach())
            loss = self.criterion_ce(pred, sid)

            acc = (pred.argmax(-1) == sid).float().mean()
            log_bar_dict = {
                'cls_loss': round(loss.item(), 3),
                'cls_acc': round(acc.item(), 3),
            }

            if should_log:
                self.log_scalars(log_bar_dict, step=self.trainer.global_step)

        self.log_dict(log_bar_dict, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sid = batch['sid']
        mel = batch['mel']
        # should_log = (self.trainer.global_step % self.trainer.log_every_n_steps == 0)
        # if should_log:
        c, _, _ = self.encoder(mel)
        c = self.act(c)
        pred = self.classifier(c.detach())

        acc = (pred.argmax(-1) == sid).float().mean()
        log_scalars = {
            'cls_acc_val': acc.item()
        }
        return log_scalars

    def validation_epoch_end(self, val_step_outputs):
        n_item = len(val_step_outputs)
        log_scalars = val_step_outputs.pop(0)
        for dct in val_step_outputs:
            for k, v in dct.items():
                log_scalars[k] += v
        for k, v in log_scalars.items():
            log_scalars[k] /= n_item
        self.log_scalars(log_scalars, step=self.trainer.global_step)

    def _model(self, x, x_cond=None):
        """
        x: (batch_size, mel_channel, seg_len)
        """
        x = rearrange(x, 'n c t -> n 1 c t')
        len_x = x.size(-1)
        if x_cond:
            x_cond = rearrange(x_cond, 'n c t -> n 1 c t')
        else:
            x_cond = torch.cat((x[..., len_x//2:], x[..., :len_x//2]), axis=-1)
        enc, mns_enc, sds_enc = self.encoder(x)
        cond, mns_cond, sds_cond = self.encoder(x_cond)
        enc = (self.act(enc), mns_enc, sds_enc)
        cond = (self.act(cond), mns_cond, sds_cond)
        y = self.decoder(enc, cond)
        return y

    def forward(self, source, target):

        original_source_len = source.size(-1)
        original_target_len = target.size(-1)
        source = rearrange(source, 'n c t -> n 1 c t')
        target = rearrange(target, 'n c t -> n 1 c t')

        if original_source_len % 8 != 0:
            source = F.pad(source, (0, 8 - original_source_len %
                                    8), mode='reflect')
        if original_target_len % 8 != 0:
            target = F.pad(target, (0, 8 - original_target_len %
                                    8), mode='reflect')

        x, x_cond = source, target

        # x = x[:,None,:,:]
        # x_cond = x_cond[:,None,:,:]

        enc, mns_enc, sds_enc = self.encoder(x)  # , mask=x_mask)
        cond, mns_cond, sds_cond = self.encoder(x_cond)  # , mask=cond_mask)

        enc = (self.act(enc), mns_enc, sds_enc)
        cond = (self.act(cond), mns_cond, sds_cond)

        y = self.decoder(enc, cond)

        dec = y[..., :original_source_len]

        return dec

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], shuffle=False, batch_size=self.batch_size, num_workers=4)

    def configure_optimizers(self):
        params_vc = list(self.encoder.parameters()) + \
            list(self.decoder.parameters())
        optimizer_vc = get_optim(params=params_vc, **self.optimizer_config)
        optimizer_cls = get_optim(
            params=self.classifier.parameters(), **self.optimizer_config)
        return [optimizer_vc, optimizer_cls], []

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items
