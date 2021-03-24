import os
import sys

import wandb
from qqdm import qqdm
import pytorch_lightning as pl

from ..util.transform import viridis


class LitProgressBar(pl.callbacks.progress.ProgressBar):
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



class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dry_run = False

    def set_dry(self):
        self.dry_run = True

    def log_func(func):
        def wrapper(*args, **kwargs):
            if args[0].dry_run:
                return
            else:
                return func(*args, **kwargs)
        return wrapper

    @log_func
    def log_mel(self, key, mel):
        self.log(key, wandb.Image(viridis(mel, flip=True)))

    @log_func
    def log_audio(self, key, audio):
        self.log(key, wandb.Audio(audio, sample_rate=22050))

    @log_func
    def log_scalars(self, data, step):
        self.logger.experiment.log(data)

    @log_func
    def log_mels(self, key, data, step):
        log_data = [
            wandb.Image(viridis(v, flip=True), caption=k) for k, v in data.items()
        ]
        self.logger.experiment.log({key: log_data})

    @log_func
    def log_audios(self, key, data, step):
        log_data = [
            wandb.Audio(v, caption=k, sample_rate=22050) for k, v in data.items()
        ]
        self.logger.experiment.log({key: log_data})
