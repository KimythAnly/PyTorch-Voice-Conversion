import os
import logging

import torch
import numpy as np
import pandas
from glob import glob
from qqdm import qqdm
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool
from ..vocoder import melgan


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def set_args(subparser):
    subparser.description = 'Preprocess melspectrogram (MelGAN).'
    subparser.add_argument('--incsv', '-i', type=str, required=True)
    subparser.add_argument('--outdir', '-o', type=str, required=True)
    subparser.add_argument('--njobs', '-j', type=int, default=8)


def check_args(args):
    if not os.path.isfile(args.incsv):
        logger.error('Input "%s" should be a CSV file.', args.incsv)
        logger.info('Bye!')
        exit(1)
    if os.path.isfile(args.outdir):
        logger.error('Output "%s" should be a directory, instead of a file.', args.outdir)
        logger.info('Bye!')
        exit(2)

    if not os.path.isdir(args.outdir):
        # logger.error('Output "%s" should be a directory', args.outdir)
        # logger.info('Bye!')
        # exit(1)
        logger.info('Output "%s" does not exist, create a directory.', args.outdir)

def preprocess_one(inp, wav2mel_fn, output_dir):
    idx, speaker, path = inp
    mel = wav2mel_fn(path)
    np.save(os.path.join(output_dir, idx+'.npy'), mel)


def main(args):
    output_dir=os.path.join(args.outdir, 'mel')
    os.makedirs(output_dir, exist_ok=True)

    vocoder = melgan.Vocoder(device='cpu')
    df = pandas.read_csv(args.incsv, sep='\t')
    
    process_list = []
    for _, data in qqdm(df.iterrows(), total=len(df)):
        # (idx, speaker, path)
        # mel = vocoder.wav2mel(path)
        process_list.append(data)

    task = partial(preprocess_one, wav2mel_fn=vocoder.wav2mel, output_dir=output_dir)
    with ThreadPool(args.njobs) as pool:
        tw = qqdm(pool.imap(task, process_list), total=len(df), desc=f'Preprocessing ')
        _ = list(tw)