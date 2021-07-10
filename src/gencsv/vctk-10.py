import os
import logging
from glob import glob

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def main(
    input_dir: str,
    output_csv: str,
):
    """
    Generate csv file for VCTK-10 preprocessing.
    """
    input_dir = os.path.realpath(input_dir)
    results = []
    speaker_dirs = glob(os.path.join(input_dir, 'train', '*'))
    for speaker_dir in speaker_dirs:
        speaker = os.path.basename(speaker_dir)
        wavs = glob(os.path.join(speaker_dir, '*.wav'))
        for wav in wavs:
            basename = os.path.basename(wav).split('.')[0]
            idx = f'{speaker}_{basename}'
            results.append(f'{idx}\t{speaker}\t{wav}')

    with open(output_csv, 'w') as f:
        f.write('idx\tspeaker\tpath\n')
        for line in results:
            f.write(line+'\n')
