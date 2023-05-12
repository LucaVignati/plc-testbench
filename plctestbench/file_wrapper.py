from __future__ import annotations
from numpy import ndarray
import os
import soundfile as sf
import numpy as np
import pickle

class FileWrapper(object):
    def __init__(self, data, path: str, persist=True) -> None:
        self.data = data
        self.path = path
        self.persist = persist

    @classmethod
    def from_path(cls, path: str) -> FileWrapper:
        if path.split('.')[-1] == 'wav':
            file = AudioFile(path=path)
        else:
            file = DataFile(path=path)
        file.load()
        return file

    def get_data(self) -> ndarray:
        return self.data

    def set_data(self, data: ndarray) -> None:
        self.data = data
        if self.persist:
            self.save()

    def get_path(self) -> str:
        return self.path

    def set_path(self, path) -> None:
        self.path = path

    def save(self) -> None:
        pass
    
    def delete(self) -> None:
        os.remove(self.path)

class AudioFile(FileWrapper):
    def __init__(self, data: ndarray=None,
                       path: str=None,
                       samplerate: float=None,
                       channels: int=None,
                       subtype: str=None,
                       endian: str=None,
                       format: str=None,
                       persist=True) -> None:
        super().__init__(data, path, persist)

        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self.endian = endian
        self.format = format

    @classmethod
    def from_audio_file(cls, audio_file: AudioFile) -> AudioFile:
        new_instance = cls(audio_file.data,
                           audio_file.path,
                           audio_file.samplerate,
                           audio_file.channels,
                           audio_file.subtype,
                           audio_file.endian,
                           audio_file.format)
        return new_instance

    def get_samplerate(self) -> float:
        return self.samplerate

    def get_channels(self) -> int:
        return self.channels

    def get_subtype(self) -> str:
        return self.subtype

    def get_endian(self) -> str:
        return self.endian

    def get_format(self) -> str:
        return self.format

    def save(self) -> None:
        sf.write(self.path,
                 self.data,
                 self.samplerate,
                 self.subtype,
                 self.endian,
                 self.format)

    def load(self) -> ndarray:
        with sf.SoundFile(self.path, 'r') as file:
            self.data = file.read()
            self.path = file.name
            self.samplerate = file.samplerate
            self.channels = file.channels
            self.subtype = file.subtype
            self.endian = file.endian
            self.format = file.format
        
        return self.data

class DataFile(FileWrapper):
    def __init__(self, data=None, path: str=None, persist=True) -> None:
        super().__init__(data, path, persist)
        if self.persist:
            self.save()

    def save(self) -> None:
        with open(self.path, 'wb') as file:
            pickle.dump(self.data, file)

    def load(self) -> None:
        with open(self.path, 'rb') as file:
            self.data = pickle.load(file)

class OutputAnalysis():
    pass

class MSEData(OutputAnalysis):
    def __init__(self, mse: ndarray) -> None:
        self._mse = np.array(mse)

    def get_mse(self) -> ndarray:
        return self._mse

class PEAQData(OutputAnalysis):
    def __init__(self, peaq_odg: float, peaq_di: float) -> None:
        self._peaq_odg = peaq_odg
        self._peaq_di = peaq_di

    def get_odg(self) -> float:
        return self._peaq_odg

    def get_di(self) -> float:
        return self._peaq_di