import os
import logging
import importlib
from glob import glob

import fire

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


INVALID_MODULES = {'dataset', 'preprocess', 'util', 'vocoder', 'classifier'}


def get_valid_modules():
    modules = glob(os.path.join('implementation', '*'))
    valid_modules = {}
    for module in modules:
        if os.path.isdir(module):
            basename = os.path.basename(module)
            if not basename.startswith('_') and basename not in INVALID_MODULES:
                valid_modules[basename] = importlib.import_module(
                    f'.{basename}.train', 'implementation').main
    return valid_modules


def main():
    valid_modules = get_valid_modules()
    fire.Fire(valid_modules)


if __name__ == '__main__':
    main()
