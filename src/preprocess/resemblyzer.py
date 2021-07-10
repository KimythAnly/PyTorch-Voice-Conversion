import os
import logging

import numpy as np
import pandas
import librosa
from resemblyzer import VoiceEncoder
from qqdm import qqdm
from functools import partial
from multiprocessing.pool import ThreadPool


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_one(inp, voice_encoder, output_dir):
    idx, speaker, path = inp
    y, sr = librosa.load(path)
    emb = voice_encoder.embed_utterance(y)
    assert (isinstance(emb, np.ndarray))
    np.save(os.path.join(output_dir, idx+'.npy'), emb.astype(np.float32))


def main(
    input_csv: str,
    output_dir: str,
    njobs: int = 12,
):
    """
    Preprocess resemblyzer feature.
    """
    output_dir = os.path.join(output_dir, 'resemblyzer')
    os.makedirs(output_dir, exist_ok=True)

    df = pandas.read_csv(input_csv, sep='\t')
    voice_encoder = VoiceEncoder()

    process_list = []
    for _, data in qqdm(df.iterrows(), total=len(df), desc='Prepare data'):
        process_list.append(data)

    task = partial(preprocess_one, voice_encoder=voice_encoder,
                   output_dir=output_dir)
    with ThreadPool(njobs) as pool:
        tw = qqdm(pool.imap(task, process_list),
                  total=len(df), desc=f'Preprocessing ')
        _ = list(tw)
