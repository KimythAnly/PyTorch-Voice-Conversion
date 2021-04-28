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


def main(config, seed=961998, dry_run=False):
    """
    AdaIN-VC: https://arxiv.org/abs/1904.05742q
    """
    pl.utilities.seed.seed_everything(seed=seed)

    logger.info('load_config()')
    config = load_config(config)
    
    name = config.pop('name')
    model_config = config.pop('model')
    dataset_config = config.pop('dataset')
    ckpt_dir = config.pop('ckpt_dir', 'checkpoints')
    trainer_config = config.pop('trainer')
    save_every_n_train_steps = trainer_config.pop('save_every_n_train_steps')

    logger.info('get_dataset()')
    dataset = get_dataset(**dataset_config)

    bar_callback = LitProgressBar()

    model = Model(
        dataset=dataset,
        **model_config,
    )

    if dry_run:
        lit_logger = None
        model.set_dry()
    else:
        lit_logger = WandbLogger(name=name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, name),
        filename=name + '-{step}',
        every_n_train_steps=save_every_n_train_steps,
    )

    trainer = pl.Trainer(
        callbacks=[bar_callback, checkpoint_callback],
        logger=lit_logger,
        **trainer_config,   
    )

    trainer.fit(model)
