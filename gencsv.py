import logging
import argparse
from pathlib import Path

from src.gencsv import get_valid_modules

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args(valid_commands):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=valid_commands)
    parser.add_argument('--input-dir', '-i', type=Path, required=True)
    parser.add_argument('--output-csv', '-o', type=Path, required=True)
    return parser.parse_args()


def main():
    valid_modules = get_valid_modules()
    args = get_args(valid_commands=valid_modules)
    main_fn = valid_modules[args.command]
    main_fn(input_dir=args.input_dir, output_csv=args.output_csv)

if __name__ == '__main__':
    main()
