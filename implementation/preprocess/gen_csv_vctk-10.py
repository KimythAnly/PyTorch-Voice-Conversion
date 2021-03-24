import os
import logging
from glob import glob
import argparse

from ..util.parser import get_parser

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def set_args(subparser):
    subparser.description = 'Generate CSV file for VCTK-10 preprocessing.'
    subparser.add_argument('--indir', '-i', type=str, required=True)
    subparser.add_argument('--outcsv', '-o', type=str, required=True)


def check_args(args):
    if not os.path.isdir(args.indir):
        logger.error('Input "%s" should be a directory.', args.indir)
        logger.info('Bye!')
        exit(1)
    if os.path.isfile(args.outcsv):
        logger.info('Output "%s" exists, overwrite it.', args.outcsv)


def main(args):
    results = []
    speaker_dirs = glob(os.path.join(args.indir, 'train', '*'))
    # speaker_dirs += glob(os.path.join(args.indir, 'test', '*'))
    for speaker_dir in speaker_dirs:
        speaker = os.path.basename(speaker_dir)
        wavs = glob(os.path.join(speaker_dir, '*.wav'))
        for wav in wavs:
            basename = os.path.basename(wav).split('.')[0]
            idx = f'{speaker}_{basename}'
            results.append(f'{idx}\t{speaker}\t{wav}')
    
    with open(args.outcsv, 'w') as f:
        f.write('idx\tspeaker\tpath\n')
        for line in results:
            f.write(line+'\n')