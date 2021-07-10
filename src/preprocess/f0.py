import os
import logging
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas
import librosa
import pyworld as pw
from qqdm import qqdm


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_one(inp, output_dir):
    idx, speaker, path = inp
    y, sr = librosa.load(path)
    f0, timeaxis = pw.harvest(y.astype(np.float64), sr)
    np.save(os.path.join(output_dir, idx+'.npy'), f0.astype(np.float32))


def main(
    input_csv: str,
    output_dir: str,
    njobs: int = 12,
):
    output_dir = os.path.join(output_dir, 'f0')
    os.makedirs(output_dir, exist_ok=True)

    df = pandas.read_csv(input_csv, sep='\t')

    process_list = []
    for _, data in qqdm(df.iterrows(), total=len(df), desc='Prepare data'):
        process_list.append(data)

    task = partial(preprocess_one, output_dir=output_dir)
    with ThreadPool(njobs) as pool:
        tw = qqdm(pool.imap(task, process_list),
                  total=len(df), desc=f'Preprocessing ')
        _ = list(tw)
