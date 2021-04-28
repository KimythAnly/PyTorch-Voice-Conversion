import os
import logging
import importlib
from glob import glob

import fire

from implementation.gen_csv import get_valid_modules

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    valid_modules = get_valid_modules()
    fire.Fire(valid_modules)


if __name__ == '__main__':
    main()