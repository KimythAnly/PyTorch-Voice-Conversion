import os
import logging
import importlib
from glob import glob

from implementation.util.parser import get_parser, CustomParser

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)-s %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = get_parser(description='Train VC model.')

    module_subparsers = parser.add_subparsers(dest='module', required=True)  
    valid_modules = get_valid_modules()
    valid_datasets = get_valid_datasets()
    for module_name in valid_modules:
        subparser = module_subparsers.add_parser(module_name)
        module = importlib.import_module(f'.{module_name}', 'implementation')
        if hasattr(module, 'set_train_args'):
            module.set_train_args(subparser)
            subhelp = subparser.description
        else:
            subhelp = ''
        choice_action = module_subparsers._ChoicesPseudoAction(module_name, (), subhelp)
        module_subparsers._choices_actions.append(choice_action)

        # Common arguments.
        subparser.add_argument(
            '--config', '-c', type=str, required=True, help='YAML configuration file.')
        subparser.add_argument(
            '--dataset', '-d', type=str, required=True, choices=valid_datasets, help=', '.join(valid_datasets))
        subparser.add_argument(
            '--train-dir', '-t', type=str, required=True, nargs='+', help='Training dir containing features.')
        subparser.add_argument(
            '--batch-size', '-bs', type=int, default=32, help='Training batch size.')
        subparser.add_argument(
            '--seed', type=int, default=961998, help='Random seed')
    
    args = parser.parse_args()
    return args


def get_valid_modules():
    invalid_modules = ['dataset', 'preprocess', 'util', 'vocoder']
    modules = glob(os.path.join('implementation', '*'))
    valid_modules = []
    for module in modules:
        if os.path.isdir(module):
            basename = os.path.basename(module)
            if not basename.startswith('_') and basename not in invalid_modules:
                valid_modules.append(basename)
    return valid_modules


def get_valid_datasets():
    datasets = glob(os.path.join('implementation', 'dataset', '*.py'))
    valid_datasets = []
    for dataset in datasets:
        basename = os.path.basename(dataset).split('.')[0]
        if not basename.startswith('_'):
            valid_datasets.append(basename)
    return valid_datasets


if __name__ == '__main__':
    args = get_args()

    logger.info('Loading VC system from "implementation/%s.py"', args.module)
    module = importlib.import_module(f'.{args.module}.train', 'implementation')

    # logger.info('Input: "%s"', args.inp)
    # logger.info('Output: "%s"', args.out)

    logger.info('Run check_args() of "%s"', args.module)
    module.check_args(args)  

    logger.info('Run main() of "%s"', args.module)
    module.main(args)
    logger.info('Done.')
