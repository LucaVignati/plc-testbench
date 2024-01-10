import sys
import hashlib
from time import sleep
import numpy as np

def _is_notebook() -> bool:
    '''
    This function returns True if the code is running in a Jupyter notebook.
    '''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# Conditional import of tqdm.
# The progress_monitor alias is used to display progress bars
# and it can be overridden by the user before running the testbench.
if _is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
progress_monitor = lambda caller: tqdm


def get_class(class_name):
    '''
    This function returns the class with the given name.
    '''
    for module_name, module in sys.modules.items():
        if module_name.startswith('plctestbench'):
            if hasattr(module, class_name):
                return getattr(module, class_name)
    raise ValueError(f"The class {class_name} does not exist.")

def compute_hash(obj):
    '''
    This function returns the hash of the given object.
    '''
    return int.from_bytes(hashlib.md5(str(obj).encode('utf-8')).digest()[:7], 'little')

def escape_email(email):
    '''
    This function escapes the given email address.
    '''
    return email.replace('@', '_at_').replace('.', '_dot_')

def dummy_progress_bar(worker):
    '''
    This function is used to create a dummy progress bar.
    '''
    for _ in worker.progress_monitor(range(10), desc=str(worker)):
        sleep(0.1)

def recursive_split_audio(audio: np.ndarray, xovers: list, bands: list = []) -> list:
        lp_audio, hp_audio = xovers[0].split(audio)
        bands.append(lp_audio)
        if len(xovers) == 1:
            bands.append(hp_audio)
        else:
            recursive_split_audio(hp_audio, xovers[1:], bands)
        return bands

def force_2d(arr):
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=-1)
    return arr