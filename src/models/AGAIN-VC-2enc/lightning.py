from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange, reduce

from .model import Encoder, Decoder, Activation, SpeakerEncoder
from ..classifier.model import Classifier
from ...vocoder import get_vocoder
from ...util.lit import BaseModel
from ...util.torch import get_optim


class Model(BaseModel):
    def __init__(
        self,
        dataset,
        batch_size,
        encoder,
        speaker,
        decoder,
        activation,
        optimizer,
        vocoder,
        classifier=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # Model
        self.encoder = Encoder(**encoder)
        self.speaker_encoder = SpeakerEncoder(**speaker)
        self.decoder = Decoder(**decoder)
        self.act = Activation(**activation)
        # Optimizer
        self.optimizer_config = optimizer
        # Vocoder
        self.vocoder = get_vocoder(device=self.device, **vocoder)
        # Classifier
        if classifier:
            self.train_classifier = True
            self.classifier = Classifier(**classifier.pop('model'))
            self.optimizer_classifier_config = classifier.pop('optimizer')
            self.criterion_ce = nn.CrossEntropyLoss()
        else:
            self.train_classifier = False
        # criterion
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop. It is independent of forward
        sid = batch['sid']
        mel = batch['mel']

        if optimizer_idx == 0:
            if self._train_classifier_only:
                return
            rec = self._model(mel)
            loss = self.criterion_l1(rec, mel)
            if self.should_log:
                with torch.no_grad():
                    dec = self.forward(mel[[0]], mel[[1]])
                    log_mels = {
                        'src': mel[0],
                        'tgt': mel[1],
                        'dec': dec[0],
                        'rec': rec[0],
                    }
                    self.vocoder.mel2plt(mel[0])
                    log_mels_plt = {
                        k: self.vocoder.mel2plt(v) for k, v in log_mels.items()
                    }
                    log_audios = {
                        k: self.vocoder.mel2wav(v) for k, v in log_mels.items()
                    }
                    log_scalars = {
                        'rec_loss': loss
                    }

                self.log_scalars(log_scalars, step=self.step)
                self.log_mels('train/mel', log_mels_plt,
                              step=self.step)
                self.log_audios('train/audio', log_audios,
                                step=self.step)
            log_bar_dict = {
                'rec_loss': round(loss.item(), 3),
            }

        if optimizer_idx == 1:
            if not self.train_classifier:
                return
            c, _, _ = self.encoder(mel)
            c = self.act(c)
            pred = self.classifier(c.detach())
            loss = self.criterion_ce(pred, sid)

            acc = (pred.argmax(-1) == sid).float().mean()
            log_bar_dict = {
                'cls_loss': round(loss.item(), 3),
                'cls_acc': round(acc.item(), 3),
                'steps': self.step,
            }

            if self.should_log:
                self.log_scalars(log_bar_dict.copy(), step=self.step)

        self.log_dict(log_bar_dict, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
        if not self.train_classifier:
            return {}
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

    def _model(self, x, x_cond=None):
        """
        x: (batch_size, mel_channel, seg_len)
        """
        # x = rearrange(x, 'n c t -> n 1 c t')
        len_x = x.size(-1)
        if not x_cond:
            x_cond = torch.cat((x[..., len_x//2:], x[..., :len_x//2]), axis=-1)

        enc, mns_enc, sds_enc = self.encoder(x)
        cond, mns_cond, sds_cond = self.speaker_encoder(x)
        enc = (self.act(enc), mns_enc, sds_enc)
        cond = (self.act(cond), mns_cond, sds_cond)

        y = self.decoder(enc, cond)
        return y

    def forward(self, source, target):

        original_source_len = source.size(-1)
        original_target_len = target.size(-1)

        if original_source_len % 8 != 0:
            source = F.pad(source, (0, 8 - original_source_len %
                                    8), mode='reflect')
        if original_target_len % 8 != 0:
            target = F.pad(target, (0, 8 - original_target_len %
                                    8), mode='reflect')

        x, x_cond = source, target

        enc, mns_enc, sds_enc = self.encoder(x)
        cond, mns_cond, sds_cond = self.speaker_encoder(x_cond)

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
        if self._train_classifier_only:
            optimizer_cls = get_optim(self.classifier.parameters(), self.optimizer_classifier_config)
            return [optimizer_cls, optimizer_cls]

        params_vc = list(self.encoder.parameters()) + \
            list(self.speaker_encoder.parameters()) + \
            list(self.decoder.parameters())
        optimizer_vc = get_optim(params_vc, self.optimizer_config)

        if self.train_classifier:
            optimizer_cls = get_optim(self.classifier.parameters(), self.optimizer_classifier_config)
            return [optimizer_vc, optimizer_cls], []
        else:
            return [optimizer_vc, optimizer_vc]
        
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items
