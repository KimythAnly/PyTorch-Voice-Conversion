import os
import logging
import argparse
from re import T
import tempfile
import shutil
from glob import glob
from typing import Union
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.preprocess import get_valid_modules as get_valid_preprocess_modules
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
    parser.add_argument('--source', '-s', type=Path, required=True)
    parser.add_argument('--target', '-t', type=Path, required=True)
    parser.add_argument('--output', '-o', type=Path)
    parser.add_argument('--njobs', '-j', type=int, default=12)
    args = parser.parse_args()
    return args


def get_latest_checkpoint(dirpath):
    ckpts = glob(os.path.join(dirpath, '*.ckpt'))
    latest_ckpt = max(ckpts, key=os.path.getctime)
    return latest_ckpt


def main(
    source,
    target,
    output,
    njobs,
    features,
    name,
    seed,
    dataset,
    vocoder,
    trainer,
    model,
):
    fd, preprocess_csv = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as fp:
        fp.write('idx\tspeaker\tpath\n')
        fp.write('source_idx\tsource_speaker\t%s\n' % source)
        fp.write('target_idx\ttarget_speaker\t%s\n' % target)
    
    preprocess_dir = tempfile.mkdtemp()

    valid_preprocess_modules = get_valid_preprocess_modules()
    for feature in features:
        preprocess_fn = valid_preprocess_modules[feature]
        preprocess_fn(input_csv=preprocess_csv, output_dir=preprocess_dir, njobs=njobs)

    valid_modules = get_valid_modules()
    model_name = model.pop('name')

    logger.info('Valid modules: ')
    for valid_module in valid_modules:
        logger.info('   %s', valid_module)
    assert model_name in valid_modules, '"%r" is an invalid model.' % model_name
    Model = valid_modules[model_name].Model

    pl.utilities.seed.seed_everything(seed=seed)
    
    dataset = None
    
    ckpt_dir = trainer.pop('ckpt_dir')
    ckpt_dir = os.path.join(ckpt_dir, name)

    ckpt_path = get_latest_checkpoint(dirpath=ckpt_dir)
    logger.info('Load checkpoint from "%s".' % ckpt_path)
    model = Model.load_from_checkpoint(
                ckpt_path, 
                dataset=dataset, 
                vocoder=vocoder, 
                **model)
    
    source_idx = 'source_idx'
    target_idx = 'target_idx'
    if output:
        output_path = output
    else:
        output_path = 'output.wav'
    model.inference(
        source=source_idx,
        target=target_idx,
        feature_dir=preprocess_dir,
        output_path=output_path)
    os.remove(preprocess_csv)
    shutil.rmtree(preprocess_dir, ignore_errors=True)



if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)
    main(
        source=args.source,
        target=args.target,
        output=args.output,
        njobs=args.njobs,
        **config
    )
