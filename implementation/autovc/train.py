import os
import logging
import importlib

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .lightning import Model
from ..util.lit import LitProgressBar
from ..util.config import load_config

logger = logging.getLogger(__name__)


def set_args(subparser):
    subparser.add_argument('--max-steps', type=int, default=100000)
    subparser.add_argument('--save-steps', type=int, default=10000)
    subparser.add_argument('--max-data', type=int, default=None)
    subparser.add_argument('--log-steps', type=int, default=500)
    subparser.add_argument('--ckpt-dir', type=str, default='checkpoints')
    subparser.add_argument('--dry-run', '--dry', action='store_true')


def check_args(args):
    pass

def get_dataset(dataset_name, train_dir, seg_len, n_speaker=None, n_data_per_speaker=None, max_data=None):
    dataset_module = importlib.import_module(f'.dataset.{dataset_name}', 'implementation')
    dataset = dataset_module.get_dataset(
        train_dir=train_dir,
        seg_len=seg_len,
        n_speaker=n_speaker,
        n_data_per_speaker=n_data_per_speaker,
        max_data=max_data,
    )
    return dataset

def main(args):
    pl.utilities.seed.seed_everything(seed=args.seed)

    logger.info('load_config()')
    config = load_config(args.config)
    name = config.pop('name')
    model_config = config.pop('model')

    logger.info('get_dataset()')
    dataset = get_dataset(
        dataset_name=args.dataset,
        train_dir=args.train_dir,
        seg_len=config.pop('seg_len'),
        max_data=args.max_data,
        n_speaker=config.pop('n_speaker', None),
        n_data_per_speaker=config.pop('n_data_per_speaker', None),
    )

    bar_callback = LitProgressBar()
    
    model = Model(
        dataset=dataset,
        batch_size=args.batch_size,
        generator_config=model_config.pop('generator'),
        optimizer_config=config.pop('optimizer'),
        classifier_config=config.pop('classifier'),
    )

    if args.dry_run:
        lit_logger = None
        model.set_dry()
    else:
        lit_logger = WandbLogger(name=name)

    checkpoint_callback  = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, name),
        filename=name + '-{step}',
        every_n_train_steps=args.save_steps,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_steps=args.max_steps,
        callbacks=[bar_callback, checkpoint_callback],
        logger=lit_logger,
        log_every_n_steps=args.log_steps,
    )


    trainer.fit(model)