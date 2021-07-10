import os
import sys

import wandb
from qqdm import qqdm
import pytorch_lightning as pl

from ..util.transform import viridis


class LitProgressBar(pl.callbacks.progress.ProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=5)

    def init_train_tqdm(self):
        has_main_bar = self.main_progress_bar is not None
        bar = qqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar

# class LitLogger(pl.loggers.wandb.WandbLogger):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
    


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dry_run = False
        self.step = 0
        self._train_classifier_only = False


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def is_last(self):
        if self.trainer.max_steps != None:
            return self.step == self.trainer.max_steps-1
        return False

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.should_log = self.step % self.trainer.log_every_n_steps == 0 \
            or self.is_last()
        return super().on_train_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.step += 1
        return super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

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
            log_scalars[k] = round(log_scalars[k], 3)
        
        self.log_dict(log_scalars, prog_bar=True, logger=False)

    def set_dry(self):
        self.dry_run = True

    def log_func(func):
        def wrapper(*args, **kwargs):
            if args[0].dry_run:
                return
            else:
                return func(*args, **kwargs)
        return wrapper
    
    def train_classifier_only(self):
        self._train_classifier_only = True

    @log_func
    def log_mel(self, key, mel):
        self.log(key, wandb.Image(viridis(mel, flip=True)))

    @log_func
    def log_audio(self, key, audio):
        self.log(key, wandb.Audio(audio, sample_rate=22050))

    @log_func
    def log_scalars(self, data, step):
        self.logger.experiment.log(data, step=step)

    @log_func
    def log_mels(self, key, data, step):
        log_data = [
            wandb.Image(viridis(v, flip=True), caption=k) for k, v in data.items()
        ]
        self.logger.experiment.log({key: log_data}, step=step)

    @log_func
    def log_audios(self, key, data, step):
        log_data = [
            wandb.Audio(v, caption=k, sample_rate=22050) for k, v in data.items()
        ]
        self.logger.experiment.log({key: log_data}, step=step)
