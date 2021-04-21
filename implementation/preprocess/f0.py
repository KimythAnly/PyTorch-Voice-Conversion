import os
import logging

import numpy as np
import pandas
import librosa
import pyworld as pw
from glob import glob
from qqdm import qqdm
from functools import partial
from multiprocessing.pool import ThreadPool


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def set_args(subparser):
    subparser.description = 'Preprocess f0.'
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

def preprocess_one(inp, output_dir):
    idx, speaker, path = inp
    y, sr = librosa.load(path)
    f0, timeaxis = pw.harvest(y.astype(np.float64), sr)
    np.save(os.path.join(output_dir, idx+'.npy'), f0.astype(np.float32))

def main(args):
    output_dir=os.path.join(args.outdir, 'f0')
    os.makedirs(output_dir, exist_ok=True)

    df = pandas.read_csv(args.incsv, sep='\t')
    
    process_list = []
    for _, data in qqdm(df.iterrows(), total=len(df), desc='Prepare data'):
        process_list.append(data)

    task = partial(preprocess_one, output_dir=output_dir)
    with ThreadPool(args.njobs) as pool:
        tw = qqdm(pool.imap(task, process_list), total=len(df), desc=f'Preprocessing ')
        _ = list(tw)