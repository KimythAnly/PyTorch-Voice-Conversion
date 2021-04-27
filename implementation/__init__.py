import os
import importlib
from glob import glob

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
