import os
import importlib
from glob import glob

INVALID_MODULES = {
    'AdaIN-VC',
    'AutoVC',
    'VQVC+',
    'classifier',
}


def get_valid_modules():
    modules = glob(os.path.join(os.path.dirname(__file__), '*'))
    valid_modules = {}
    for module in modules:
        if os.path.isdir(module):
            # basename = os.path.basename(module).split('.')[0]
            basename = os.path.basename(module)
            if not basename.startswith('_') and basename not in INVALID_MODULES:
                valid_modules[basename] = importlib.import_module(
                    f'.{basename}.lightning', __package__)
    return valid_modules