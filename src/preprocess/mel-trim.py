import os
import logging
from functools import partial

import numpy as np
import pandas
from qqdm import qqdm
from functools import partial
from multiprocessing.pool import ThreadPool
from ..vocoder import melgan


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_one(inp, wav2mel_fn, output_dir):
    idx, speaker, path = inp
    mel = wav2mel_fn(path)
    np.save(os.path.join(output_dir, idx+'.npy'), mel)


def main(
    input_csv: str,
    output_dir: str,
    njobs: int = 12,
):
    """
    Preprocess trimmed melspectrogram for MelGAN.
    """
    output_dir = os.path.join(output_dir, 'mel-trim')
    os.makedirs(output_dir, exist_ok=True)

    vocoder = melgan.Vocoder(device='cpu')
    df = pandas.read_csv(input_csv, sep='\t')

    process_list = []
    for _, data in qqdm(df.iterrows(), total=len(df), desc='Create data list'):
        # (idx, speaker, path)
        # mel = vocoder.wav2mel(path)
        process_list.append(data)

    wav2mel_fn = partial(vocoder.wav2mel, trim=15)

    task = partial(preprocess_one, wav2mel_fn=wav2mel_fn,
                   output_dir=output_dir)
    with ThreadPool(njobs) as pool:
        tw = qqdm(pool.imap(task, process_list),
                  total=len(df), desc=f'Preprocessing ')
        _ = list(tw)
