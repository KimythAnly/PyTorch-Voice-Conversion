import os
import logging
import argparse
from typing import Union
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import get_valid_modules
from src.dataset import get_dataset
from src.util.lit import LitProgressBar
from src.util.config import load_config


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=Path, required=True)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    return args


def main(
    name,
    seed,
    dataset,
    vocoder,
    trainer,
    model,
    **kwargs,
):
    valid_modules = get_valid_modules()
    model_name = model.pop('name')
    logger.info('Valid modules: ')
    for valid_module in valid_modules:
        logger.info('   %s', valid_module)
    assert model_name in valid_modules, '"%r" is an invalid model.' % model_name
    Model = valid_modules[model_name].Model

    pl.utilities.seed.seed_everything(seed=seed)
    
    logger.info('Getting dataset: %s', dataset)
    
    dataset = get_dataset(**dataset)
    
    ckpt_dir = trainer.pop('ckpt_dir')
    os.makedirs(os.path.join(ckpt_dir, name), exist_ok=True)
    ckpt_dir = os.path.join(ckpt_dir, name)

    logger.info('Saving speaker name dict and config dict to "%s"', ckpt_dir)

    # save_dict(
    #     data=config_dict,
    #     filename=os.path.join(ckpt_dir, 'config.yaml')
    # )

    bar_callback = LitProgressBar()

    model = Model(
        dataset=dataset,
        vocoder=vocoder,
        **model,
    )

    if args.dry:
        model.set_dry()


    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=name + '-{step}',
        every_n_train_steps=trainer.pop('save_every_n_train_steps'),
    )

    trainer = pl.Trainer(
        callbacks=[bar_callback, checkpoint_callback],
        logger=lit_logger,
        **trainer,
    )
    trainer.fit(model)


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)
    if args.dry:
        lit_logger = None
    else:
        lit_logger = WandbLogger(name=config['name'])
        lit_logger.log_hyperparams(config)

    main(**config)
