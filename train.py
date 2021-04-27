import os
import logging

import fire

from implementation import get_valid_modules

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    valid_modules = get_valid_modules()
    fire.Fire(valid_modules)


if __name__ == '__main__':
    main()
