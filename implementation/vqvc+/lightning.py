from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange, reduce

from .model import VC_MODEL
from ..classifier.model import Classifier
from ..util.lit import BaseModel
from ..util.torch import get_optim
from ..vocoder.melgan import Vocoder


class Model(BaseModel):
    def __init__(
        self,
        dataset,
        batch_size,
        model,
        optimizer,
        classifier,
        loss,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # Model
        self.model = VC_MODEL(**model)
        # Optimizer
        self.optimizer_config = optimizer
        # Vocoder
        self.vocoder = Vocoder(device=self.device)
        # criterion
        self.criterion_l1 = nn.L1Loss()
        # Classifier
        if classifier:
            self.train_classifier = True
            self.classifier = Classifier(**classifier.pop('model'))
            self.optimizer_classifier_config = classifier.pop('optimizer')
            self.criterion_ce = nn.CrossEntropyLoss()
        else:
            self.train_classifier = False
        self.loss_config = loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # training_step defined the train loop. It is independent of forward
        sid = batch['sid']
        mel = batch['mel']

        should_log = (self.trainer.global_step + 1) \
            % self.trainer.log_every_n_steps == 0

        if optimizer_idx == 0:
            rec, dec, enc_content, spk, latent_loss, idx = self.model(mel)

            rec_loss = self.criterion_l1(rec, mel)
            latent_loss = latent_loss.mean()

            loss = rec_loss + self.loss_config['lambda_latent'] * latent_loss

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
                        'rec_loss': rec_loss,
                        'latent_loss': latent_loss,
                    }

                self.log_scalars(log_scalars, step=self.trainer.global_step)
                self.log_mels('train/mel', log_mels_plt,
                              step=self.trainer.global_step)
                self.log_audios('train/audio', log_audios,
                                step=self.trainer.global_step)
            log_bar_dict = {
                'rec_loss': round(rec_loss.item(), 3),
                'latent_loss': round(latent_loss.item(), 3),
                'total_steps': int(self.trainer.global_step),
            }

        if optimizer_idx == 1:
            c = self.get_content(mel)
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
        if not self.train_classifier:
            return
        sid = batch['sid']
        mel = batch['mel']
        # should_log = (self.trainer.global_step % self.trainer.log_every_n_steps == 0)
        # if should_log:
        c = self.get_content(mel)
        pred = self.classifier(c.detach())

        acc = (pred.argmax(-1) == sid).float().mean()
        log_scalars = {
            'cls_acc_val': acc.item()
        }
        return log_scalars

    def validation_epoch_end(self, val_step_outputs):
        if not self.train_classifier:
            return
        n_item = len(val_step_outputs)
        log_scalars = val_step_outputs.pop(0)
        for dct in val_step_outputs:
            for k, v in dct.items():
                log_scalars[k] += v
        for k, v in log_scalars.items():
            log_scalars[k] /= n_item
        self.log_scalars(log_scalars, step=self.trainer.global_step)

    def get_content(self, x):
        c, _, _, _ = self.model.encode(x)
        return c[0]


    def forward(self, source, target):

        original_source_len = source.size(-1)
        original_target_len = target.size(-1)

        if original_source_len % 8 != 0:
            source = F.pad(source, (0, 8 - original_source_len %
                                    8), mode='reflect')
        if original_target_len % 8 != 0:
            target = F.pad(target, (0, 8 - original_target_len %
                                    8), mode='reflect')

        # emb = self.speaker_encoder(x_cond)
        # mu, log_sigma = self.content_encoder(x)
        # y = self.decoder(mu, emb)
        dec = self.model.inference(source, target)

        dec = dec[..., :original_source_len]

        return dec

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        if self.dataset['val']:
            return DataLoader(self.dataset['val'], shuffle=False, batch_size=self.batch_size, num_workers=4)
        else:
            return None

    def configure_optimizers(self):
        params_vc = self.model.parameters()
        optimizer_vc = get_optim(params_vc, self.optimizer_config)
        if self.train_classifier:
            optimizer_cls = get_optim(self.classifier.parameters(), self.optimizer_classifier_config)
            return [optimizer_vc, optimizer_cls], []
        else:
            return optimizer_vc

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items
