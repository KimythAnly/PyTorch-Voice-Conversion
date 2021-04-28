import os
import importlib
from glob import glob

INVALID_MODULES = {}


def get_valid_modules():
    modules = glob(os.path.join(os.path.dirname(__file__), '*.py'))
    valid_modules = {}
    for module in modules:
        if os.path.isfile(module):
            basename = os.path.basename(module).split('.')[0]
            if not basename.startswith('_') and basename not in INVALID_MODULES:
                valid_modules[basename] = importlib.import_module(
                    f'.{basename}', __package__).main
    return valid_modules