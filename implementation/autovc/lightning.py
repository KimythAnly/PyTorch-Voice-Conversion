from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange, reduce

from .model import Generator
from ..classifier.model import Classifier
from ..util.lit import BaseModel
from ..util.torch import get_optim
from ..vocoder.melgan import Vocoder


class Model(BaseModel):
    def __init__(
        self,
        dataset,
        batch_size,
        generator_params,
        optimizer_config,
        classifier_config,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # Model
        self.generator = Generator(**generator_params)
        # Optimizer
        self.optimizer_config = optimizer_config
        # Vocoder
        self.vocoder = Vocoder(device=self.device)
        # Classifier
        if classifier_config:
            self.train_classifier = True
            self.classifier = Classifier(**classifier_config.pop('model'))
            self.optimizer_classifier_config = classifier_config.pop('optimizer')
            self.criterion_ce = nn.CrossEntropyLoss()
        else:
            self.train_classifier = False
        # criterion
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop. It is independent of forward
        sid = batch['sid']
        mel = batch['mel']
        emb = batch['emb']

        should_log = self.step % self.trainer.log_every_n_steps == 0

        if optimizer_idx == 0:
            x_identic, x_identic_psnt, code_real = self._model(mel, emb, emb)
        
            loss_id = self.criterion_mse(x_identic, mel)
            loss_id_psnt = self.criterion_mse(x_identic_psnt, mel)
            loss = loss_id + loss_id_psnt
            loss_l1 = self.criterion_l1(x_identic_psnt, mel)


            if should_log:
                with torch.no_grad():
                    rec = self.forward(mel[[0]], emb[[0]], emb[[0]])
                    dec = self.forward(mel[[0]], emb[[0]], emb[[1]])
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
                        'rec_loss': loss_l1
                    }

                self.log_scalars(log_scalars, step=self.step)
                self.log_mels('train/mel', log_mels_plt,
                              step=self.step)
                self.log_audios('train/audio', log_audios,
                                step=self.step)
            log_bar_dict = {
                'rec_loss': round(loss_l1.item(), 3),
            }

        if optimizer_idx == 1:
            c = self._model(mel, emb, None)
            pred = self.classifier(c.detach())
            loss = self.criterion_ce(pred, sid)

            acc = (pred.argmax(-1) == sid).float().mean()
            log_bar_dict = {
                'cls_loss': round(loss.item(), 3),
                'cls_acc': round(acc.item(), 3),
                'steps': self.step,
            }

            if should_log:
                self.log_scalars(log_bar_dict, step=self.step)

        self.log_dict(log_bar_dict, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sid = batch['sid']
        mel = batch['mel']
        emb = batch['emb']
        # should_log = (self.trainer.global_step % self.trainer.log_every_n_steps == 0)
        # if should_log:
        c = self._model(mel, emb, None)
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
        self.log_scalars(log_scalars, step=self.step)

    def _model(self, x_real, emb_org, emb_trg):
        x_real = rearrange(x_real, 'n c t -> n t c')
        x_identic, x_identic_psnt, code_real = self.generator(x_real, emb_org, emb_trg)
        code_real = rearrange(code_real, 'n t c -> n c t')
        if emb_trg is None:
            return code_real
        else:
            dec = rearrange(x_identic, 'n 1 t c -> n c t')
            dec_psnt = rearrange(x_identic_psnt, 'n 1 t c -> n c t')
            return dec, dec_psnt, code_real

    def forward(self, x_real, emb_org, emb_trg):
        x_real = rearrange(x_real, 'n c t -> n t c')
        original_source_len = x_real.size(-2)

        if original_source_len % 32 != 0:
            x_real = F.pad(x_real.transpose(1,2), (0, 32 - original_source_len % 32), mode='reflect').transpose(1,2)

        x_identic, x_identic_psnt, code_real = self.generator(x_real, emb_org, emb_trg)
        dec = rearrange(x_identic_psnt, 'n 1 t c -> n c t')
        return dec


    def train_dataloader(self):
        return DataLoader(self.dataset['train'], shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], shuffle=False, batch_size=self.batch_size, num_workers=4)

    def configure_optimizers(self):
        params_vc = list(self.generator.parameters())
        optimizer_vc = get_optim(params_vc, self.optimizer_config)
        if self.train_classifier:
            optimizer_cls = get_optim(self.classifier.parameters(), self.optimizer_classifier_config)
            return [optimizer_vc, optimizer_cls], []
        else:
            return optimizer_vc

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        items.pop('v_num', None)
        return items
