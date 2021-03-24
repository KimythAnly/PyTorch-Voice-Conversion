import os
import logging
import importlib
from glob import glob

from implementation.util.parser import get_parser

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = get_parser(description='Preprocess')
    subparsers = parser.add_subparsers(dest='module', required=True)
    valid_modules = get_valid_modules()
    for module_name in valid_modules:
        subparser = subparsers.add_parser(module_name)
        module = importlib.import_module(f'.preprocess.{module_name}', 'implementation')
        if hasattr(module, 'set_args'):
            module.set_args(subparser)
            subhelp = subparser.description

        choice_action = subparsers._ChoicesPseudoAction(module_name, (), subhelp)
        subparsers._choices_actions.append(choice_action)
    args = parser.parse_args()
    return args

def get_valid_modules():
    modules = glob(os.path.join('implementation', 'preprocess', '*.py'))
    valid_modules = []
    for module in modules:
        basename = os.path.basename(module).split('.')[0]
        if not basename.startswith('_'):
            valid_modules.append(basename)
    return valid_modules

if __name__ == '__main__':
    args = get_args()

    logger.info('Loading module from "implementation/preprocess/%s.py"', args.module)
    module = importlib.import_module(f'.preprocess.{args.module}', 'implementation')

    logger.info('Run check_args() of "%s"', args.module)
    module.check_args(args)

    logger.info('Run main() of "%s"', args.module)
    module.main(args)

    logger.info('Done.')
