from anytree import NodeMixin
import numpy as np
from ecctestbench.file_wrapper import AudioFile, DataFile

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    def __init__(self, file=None, worker=None, absolute_path=None, parent=None) -> None:
        self.file = file
        self.worker = worker
        self.parent = parent
        self.folder_name = None
        self.absolute_path = absolute_path

    def set_folder_name(self, folder_name) -> None:
        self.folder_name = folder_name

    def set_file(self, file) -> None:
        self.file = file

    def get_file(self):
        return self.file
    
    def get_folder_name(self):
        return self.folder_name

    def set_path(self, absolute_path):
        self.absolute_path = absolute_path

    def get_worker(self):
        return self.worker

    def get_original_track(self):
        return self.root.get_file()

    def get_lost_samples_mask(self):
        return self.ancestors[1].get_file()

    def get_ecc_track(self):
        return self.ancestors[2].get_file()
    
    def __str__(self) -> str:
        return "file: " + str(self.file) + '\n' +\
               "worker: " + str(self.worker) + '\n' +\
               "folder name: " + str(self.folder_name) + '\n' + \
               "absolute path: " + str(self.absolute_path)

class OriginalTrackNode(Node):
    def __init__(self, file=None, worker=None, absolute_path=None, parent=None) -> None:
        super().__init__(file=file, worker=worker, absolute_path=absolute_path, parent=parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def run(self) -> None:
        pass
    
class LostSamplesMaskNode(Node):
    def __init__(self, file=None, worker=None, absolute_path=None, parent=None) -> None:
        super().__init__(file=file, worker=worker, absolute_path=absolute_path, parent=parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()
    
    def run(self) -> None:
        original_track_data = self.get_original_track().get_data()
        num_samples = len(original_track_data)
        lost_samples_mask = self.get_worker().run(num_samples)
        self.file = DataFile(self.absolute_path + '.npy', lost_samples_mask)

class ECCTrackNode(Node):
    def __init__(self, file=None, worker=None, absolute_path=None, parent=None) -> None:
        super().__init__(file=file, worker=worker, absolute_path=absolute_path, parent=parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()
    
    def run(self) -> None:
        original_track = self.get_original_track()
        original_track_data = original_track.get_data()
        lost_samples_mask = self.get_lost_samples_mask().get_data()
        ecc_track = self.get_worker().run(original_track_data, lost_samples_mask)
        self.file = AudioFile.from_audio_file(original_track)
        self.file.set_path(self.absolute_path + '.wav')
        self.file.set_data(ecc_track)

class OutputAnalysisNode(Node):
    def __init__(self, file=None, worker=None, absolute_path=None, parent=None) -> None:
        super().__init__(file=file, worker=worker, absolute_path=absolute_path, parent=parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def run(self) -> None:
        original_track_data = self.get_original_track().get_data()
        ecc_track_data = self.get_ecc_track().get_data()
        output_analysis = self.get_worker().run(original_track_data, ecc_track_data)
        self.file = DataFile(self.absolute_path + '.npy', output_analysis)