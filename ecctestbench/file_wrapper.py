from __future__ import annotations
from numpy import ndarray
import soundfile as sf
import numpy as np

class FileWrapper(object):
    def __init__(self, data: ndarray, path: str, persist=True) -> None:
        self.data = data
        self.path = path
        self.persist = persist

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

class AudioFile(FileWrapper):
    def __init__(self, data: ndarray,
                       path: str,
                       samplerate: float,
                       channels: int,
                       subtype: str,
                       endian: str,
                       format: str,
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

    @classmethod
    def from_path(cls, path: str) -> AudioFile:
        with sf.SoundFile(path, 'r') as file:
            new_instance = cls(file.read(),
                               path,
                               file.samplerate,
                               file.channels,
                               file.subtype,
                               file.endian,
                               file.format)
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
    def __init__(self, data: ndarray, path: str, persist=True) -> None:
        super().__init__(data, path, persist)
        if self.persist:
            self.save()

    def save(self) -> None:
        np.save(self.path, self.data)

    def load(self) -> ndarray:
        self.data = np.load(self.path)
        return self.data