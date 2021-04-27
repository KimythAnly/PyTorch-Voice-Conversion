import os
import logging
import importlib

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .lightning import Model
from ..dataset import get_dataset
from ..util.lit import LitProgressBar
from ..util.config import load_config

logger = logging.getLogger(__name__)


def main(
    config,
    train_dir,
    dataset='mel_trim',
    ckpt_dir='checkpoints',
    batch_size=32,
    max_data=None,
    max_steps=100000,
    save_steps=10000,
    log_steps=500,
    seed=961998,
    dry_run=False,
):
    """
    AdaIN-VC: https://arxiv.org/abs/1904.05742q
    """
    print(config)
    pl.utilities.seed.seed_everything(seed=seed)

    logger.info('load_config()')
    config = load_config(config)
    name = config.pop('name')
    model_config = config.pop('model')

    logger.info('get_dataset()')
    dataset = get_dataset(
        dataset_name=dataset,
        train_dir=train_dir,
        seg_len=config.pop('seg_len'),
        max_data=max_data,
        n_speaker=config.pop('n_speaker', None),
        n_data_per_speaker=config.pop('n_data_per_speaker', None),
    )

    bar_callback = LitProgressBar()

    model = Model(
        dataset=dataset,
        batch_size=batch_size,
        content_config=model_config.pop('content'),
        speaker_config=model_config.pop('speaker'),
        decoder_config=model_config.pop('decoder'),
        optimizer_config=config.pop('optimizer'),
        classifier_config=config.pop('classifier', None),
        loss_config=config.pop('loss'),
    )

    if dry_run:
        lit_logger = None
        model.set_dry()
    else:
        lit_logger = WandbLogger(name=name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, name),
        filename=name + '-{step}',
        every_n_train_steps=save_steps,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_steps=max_steps,
        callbacks=[bar_callback, checkpoint_callback],
        logger=lit_logger,
        log_every_n_steps=log_steps,
        gradient_clip_val=config.pop('grad_norm'),
    )

    trainer.fit(model)
