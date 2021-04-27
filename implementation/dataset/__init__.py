import os
import importlib
import logging
from glob import glob


logger = logging.getLogger(__name__)

INVALID_MODULES = {}


def _get_valid_datasets():
    datasets = glob(os.path.join('implementation', 'dataset', '*.py'))
    valid_datasets = []
    for dataset in datasets:
        basename = os.path.basename(dataset).split('.')[0]
        if not basename.startswith('_') and not basename in INVALID_MODULES:
            valid_datasets.append(basename)
    return valid_datasets


def get_dataset(dataset_name, train_dir, seg_len, n_speaker=None, n_data_per_speaker=None, max_data=None):
    valid_datasets = _get_valid_datasets()
    if dataset_name not in valid_datasets:
        logger.error('Invalid dataset "%s"!', dataset_name)
        logger.info('Please use a valid dataset: %s.', valid_datasets)
        exit(1)
    dataset_module = importlib.import_module(
        f'.dataset.{dataset_name}', 'implementation')
    dataset = dataset_module.get_dataset(
        train_dir=train_dir,
        seg_len=seg_len,
        n_speaker=n_speaker,
        n_data_per_speaker=n_data_per_speaker,
        max_data=max_data,
    )
    return dataset
