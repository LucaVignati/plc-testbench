import sys
import hashlib
from time import sleep
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    DUMMY_BAR_SLEEP = 0
else:
    from tqdm import tqdm
    DUMMY_BAR_SLEEP = 0.1
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
        sleep(DUMMY_BAR_SLEEP)

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

def prepare_progress_monitor(progress_monitor) -> callable:
            def composite_progress_monitor(iterable, desc):
                    for item in iterable:
                        progress_monitor.update(1)
                        yield item
            return composite_progress_monitor

def relative_to_root(path):
    return PROJECT_ROOT.joinpath(path)

def extract_intorni(audio_file, lost_samples_idxs, intorno_size, fs, packet_size, unique=False):
    def find_boundary_indexes(lost_samples_idxs):
        boundary_indexes = []
        start_idx = None
        end_idx = None
        for idx in lost_samples_idxs:
            if start_idx is None:
                start_idx = idx
                end_idx = idx
            elif idx == end_idx + 1:
                end_idx = idx
            else:
                boundary_indexes.append((start_idx, end_idx))
                start_idx = None
            if idx == lost_samples_idxs[-1]:
                boundary_indexes.append((start_idx, end_idx))
        return boundary_indexes

    intorno_samples = float(intorno_size*fs)/1000
    audio_data = audio_file.get_data()
    boundary_indexes = find_boundary_indexes(lost_samples_idxs)
    intorni = []
    packet_idxs = []
    for start_idx, end_idx in boundary_indexes:
        center_idx = (start_idx + end_idx) // 2
        start_sample = int(center_idx - intorno_samples // 2)
        end_sample = int(center_idx + intorno_samples // 2)
        intorno = audio_data[start_sample:end_sample]
        if len(intorno) == 0:
            continue
        if unique:
            intorni.append(intorno.copy())
        else:
            intorni.append(intorno)
        packet_idxs.append(int(center_idx//packet_size))
    return packet_idxs, intorni

def force_single_loss_per_stimulus(lost_samples_idxs, fs, spacing, samples_per_packet):

    # TODO: Allow for consecutive lost packets. This solution will discard any subsequent lost packets.
    selected_idxs = []
    prev_packet_idx = -1
    
    for idx in lost_samples_idxs:
        packet_idx = idx // samples_per_packet
        if prev_packet_idx == -1 or (packet_idx - prev_packet_idx) >= spacing/1000 * fs / samples_per_packet or packet_idx == prev_packet_idx:
            selected_idxs.append(idx)
            prev_packet_idx = packet_idx
    
    return selected_idxs

def fade_in(audio, fs, fade_in_time) -> None:
    fade_in_samples = int(fade_in_time * fs / 1000)
    fade_in_window = np.linspace(0, 1, int(fade_in_samples), dtype=audio.dtype)
    if audio.ndim > 1:
        fade_in_window = fade_in_window[:, np.newaxis]
    audio[:fade_in_samples] *= fade_in_window

def fade_out(audio, fs, fade_out_time) -> None:
    fade_out_samples = int(fade_out_time * fs / 1000)
    fade_out_window = np.linspace(1, 0, int(fade_out_samples), dtype=audio.dtype)
    if audio.ndim > 1:
        fade_out_window = fade_out_window[:, np.newaxis]
    audio[-fade_out_samples:] *= fade_out_window

def leading_silence(audio, fs, silence_time) -> None:
    silence_samples = int(silence_time * fs / 1000)
    silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)
    return np.concatenate((silence, audio), axis=0)

def trailing_silence(audio, fs, silence_time) -> None:
    silence_samples = int(silence_time * fs / 1000)
    silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)
    return np.concatenate((audio, silence), axis=0)

def is_loud_enough(audio_data, audio_reference, threshold=-30):
    """
    Determines if the average loudness of the audio data is above a specified threshold.
    
    Args:
        audio_data (numpy.ndarray): Audio data in the shape of [samples, channels].
        audio_reference (numpy.ndarray): Audio data in the shape of [samples, channels].
        threshold (float): Loudness threshold in dB (default is -30 dB).
    
    Returns:
        bool: True if the average loudness is above the threshold, False otherwise.
    """
    # Calculate the RMS (root mean square) of the audio data
    rms = np.sqrt(np.mean(np.square(audio_data)))

    # Calculate the RMS (root mean square) of the audio reference
    rms_reference = np.sqrt(np.mean(np.square(audio_reference)))
    
    # Convert RMS to dB
    db = 20 * np.log10(rms)
    db_reference = 20 * np.log10(rms_reference)

    loud_enough = db > db_reference + threshold
    
    # Compare the average loudness to the threshold
    return loud_enough