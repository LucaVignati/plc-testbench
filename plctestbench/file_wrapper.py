from __future__ import annotations
import os
import pickle
from pathlib import Path
from numpy import ndarray
import soundfile as sf
import numpy as np
from plctestbench.utils import compute_hash

DEFAULT_DTYPE = 'float32'

def calculate_hash(*args) -> int:
    data = ''
    for arg in args:
        data = data + str(arg)
    return compute_hash(data)

class FileWrapper(object):
    def __init__(self, data=None, path: str=None, persist=True) -> None:
        self.data = np.ascontiguousarray(data.astype(DEFAULT_DTYPE)) if isinstance(data, np.ndarray) else data
        self.path = path
        self.persist = persist

        if self.path is None:
            raise ValueError('path must be specified')

        if self.data is not None:
            self.save()

        self.load()

        self.hash = calculate_hash(self.data.tobytes()) if isinstance(self.data, ndarray) else hash(self.data)

    @classmethod
    def from_path(cls, path: str) -> FileWrapper:
        if not Path(path).exists():
            return None

        if path.split('.')[-1] == 'wav':
            file = AudioFile(path=path)
        else:
            file = DataFile(path=path)
        return file

    def get_data(self) -> ndarray:
        return self.data

    def get_path(self) -> str:
        return self.path

    def set_path(self, path) -> None:
        self.path = path

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass

    def delete(self) -> None:
        os.remove(self.path)

    def __hash__(self):
        return self.hash

class AudioFile(FileWrapper):
    def __init__(self, data: ndarray=None,
                       path: str=None,
                       samplerate: float=None,
                       channels: int=None,
                       subtype: str=None,
                       endian: str=None,
                       audio_format: str=None,
                       persist=True) -> None:

        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self.endian = endian
        self.audio_format = audio_format
        super().__init__(data, path, persist)

    @classmethod
    def from_audio_file(cls, audio_file: AudioFile,
                             new_data: ndarray=None,
                             new_path: str=None,
                             new_samplerate: float=None,
                             new_channels: int=None,
                             new_subtype: str=None,
                             new_endian: str=None,
                             new_audio_format: str=None) -> AudioFile:
        data = audio_file.data if new_data is None else new_data
        path = audio_file.path if new_path is None else new_path
        samplerate = audio_file.samplerate if new_samplerate is None else new_samplerate
        channels = audio_file.channels if new_channels is None else new_channels
        subtype = audio_file.subtype if new_subtype is None else new_subtype
        endian = audio_file.endian if new_endian is None else new_endian
        audio_format = audio_file.audio_format if new_audio_format is None else new_audio_format
        new_instance = cls(data,
                           path,
                           samplerate,
                           channels,
                           subtype,
                           endian,
                           audio_format)
        return new_instance

    def get_samplerate(self) -> float:
        return self.samplerate

    def get_channels(self) -> int:
        return self.channels

    def get_subtype(self) -> str:
        return self.subtype

    def get_endian(self) -> str:
        return self.endian

    def get_audio_format(self) -> str:
        return self.audio_format

    def save(self) -> None:
        sf.write(self.path,
                 self.data,
                 self.samplerate,
                 self.subtype,
                 self.endian,
                 self.audio_format)

    def load(self) -> ndarray:
        with sf.SoundFile(self.path, 'r') as file:
            self.data = file.read(dtype=DEFAULT_DTYPE)
            self.path = file.name
            self.samplerate = file.samplerate
            self.channels = file.channels
            self.subtype = file.subtype
            self.endian = file.endian
            self.audio_format = file.format

        return self.data

class DataFile(FileWrapper):
    def __init__(self, data=None, path: str=None, persist=True) -> None:
        super().__init__(data, path, persist)

    def save(self) -> None:
        with open(self.path, 'wb') as file:
            pickle.dump(self.data, file)

    def load(self) -> None:
        with open(self.path, 'rb') as file:
            try:
                self.data = pickle.load(file)
            except pickle.UnpicklingError:
                self.data = None


class OutputAnalysis():
    pass

class SimpleCalculatorData(OutputAnalysis):
    def __init__(self, error: ndarray) -> None:
        self._error = np.ascontiguousarray(np.array(error).astype(DEFAULT_DTYPE))

    def get_error(self) -> ndarray:
        return self._error

    def __len__(self):
        return len(self._error)

    def __iter__(self):
        return self._error.__iter__()

    def __next__(self):
        return self._error.__next__()

    def __getitem__(self, key):
        return self._error.__getitem__(key)

    def __hash__(self) -> int:
        return calculate_hash(self._error.tobytes())

class PEAQData(OutputAnalysis):
    def __init__(self, peaq_odg: float, peaq_di: float) -> None:
        self._peaq_odg = peaq_odg
        self._peaq_di = peaq_di

    def get_odg(self) -> float:
        return self._peaq_odg

    def get_di(self) -> float:
        return self._peaq_di

    def __hash__(self) -> int:
        return calculate_hash(self._peaq_odg, self._peaq_di)
