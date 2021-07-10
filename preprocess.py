import logging
import argparse
from pathlib import Path

from src.preprocess import get_valid_modules
from src.util.config import load_config

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args(valid_features):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', '-c', type=Path)
    group.add_argument('--feature', '-f', choices=valid_features)
    parser.add_argument('--input-csv', '-i', type=Path, required=True)
    parser.add_argument('--output-dir', '-o', type=Path, required=True)
    parser.add_argument('--njobs', '-j', type=int, default=12)
    return parser.parse_args()

    
if __name__ == '__main__':
    valid_modules = get_valid_modules()
    args = get_args(valid_features=valid_modules)

    if args.config:
        config = load_config(args.config)
        features = config['features']
    else:
        features = [args.feature]
    
    for feature in features:
        main_fn = valid_modules[feature]
        main_fn(input_csv=args.input_csv, output_dir=args.output_dir, njobs=args.njobs)
