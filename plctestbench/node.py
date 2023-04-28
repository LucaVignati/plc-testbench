from anytree import NodeMixin
import numpy as np
from plctestbench.worker import Worker
from plctestbench.file_wrapper import FileWrapper, AudioFile, DataFile
import uuid as Uuid

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    def __init__(self, file: FileWrapper=None, worker=None, settings=None, absolute_path: str=None, parent=None, uuid = None) -> None:
        self.file = file
        if parent!=None:
            settings.inherit_from(parent.settings)
        self.settings = settings
        self.worker = worker(settings) if worker!=None else None
        self.parent = parent
        self.folder_name = None
        self.absolute_path = absolute_path
        self.uuid = uuid if uuid != None else str(Uuid.uuid4())

    def set_folder_name(self, folder_name) -> None:
        self.folder_name = folder_name

    def set_file(self, file) -> None:
        self.file = file

    def get_file(self) -> FileWrapper:
        return self.file
    
    def get_folder_name(self) -> str:
        return self.folder_name

    def set_path(self, absolute_path) -> None:
        self.absolute_path = absolute_path

    def get_path(self) -> str:
        return self.absolute_path

    def get_worker(self) -> Worker:
        return self.worker

    def get_original_track(self) -> AudioFile:
        return self.root.get_file()

    def get_lost_samples_mask(self) -> DataFile:
        return self.ancestors[1].get_file()

    def get_reconstructed_track(self) -> AudioFile:
        return self.ancestors[2].get_file()
    
    def __str__(self) -> str:
        return "file: " + str(self.file) + '\n' +\
               "worker: " + str(self.worker) + '\n' +\
               "folder name: " + str(self.folder_name) + '\n' + \
               "absolute path: " + str(self.absolute_path)

class OriginalTrackNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None) -> None:
        super().__init__(file, worker, settings, absolute_path, parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_track_name(self) -> str:
        return self.absolute_path.rpartition("/")[2].split(".")[0]
    
    def load(self) -> None:
        self.file = AudioFile.from_path(self.absolute_path + '.wav')
        self.file.persist = False
        self.file.load()

    def run(self) -> None:
        print(self.get_track_name())
        self.get_worker().run(self.uuid)
    
class LostSamplesMaskNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None) -> None:
        super().__init__(file, worker, settings, absolute_path, parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root
    
    def load(self) -> None:
        self.file = DataFile([], self.absolute_path + '.npy', persist=False)
        self.file.load()
    
    def run(self) -> None:
        original_track_data = self.get_original_track().get_data()
        num_samples = len(original_track_data)
        lost_samples_idx = self.get_worker().run(num_samples, self.uuid)
        self.file = DataFile(lost_samples_idx, self.absolute_path + '.npy')

class ReconstructedTrackNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None) -> None:
        super().__init__(file, worker, settings, absolute_path, parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root
    
    def get_lost_samples_mask_node(self) -> LostSamplesMaskNode:
        return self.ancestors[1]
    
    def load(self) -> None:
        self.file = AudioFile.from_path(self.absolute_path + '.wav')
        self.file.persist = False
        self.file.load()
    
    def run(self) -> None:
        original_track = self.get_original_track()
        original_track_data = original_track.get_data()
        lost_samples_idx = self.get_lost_samples_mask().get_data()
        reconstructed_track = self.get_worker().run(original_track_data, lost_samples_idx, self.uuid)
        self.file = AudioFile.from_audio_file(original_track)
        self.file.set_path(self.absolute_path + '.wav')
        self.file.set_data(reconstructed_track)

class OutputAnalysisNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None) -> None:
        super().__init__(file, worker, settings, absolute_path, parent)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root
    
    def get_lost_samples_mask_node(self) -> LostSamplesMaskNode:
        return self.ancestors[1]

    def get_reconstructed_track_node(self) -> ReconstructedTrackNode:
        return self.ancestors[2]
    
    def load(self) -> None:
        self.file = DataFile([], self.absolute_path + '.pickle', False)
        self.file.load()

    def run(self) -> None:
        original_track = self.get_original_track()
        reconstructed_track = self.get_reconstructed_track()
        output_analysis = self.get_worker().run(original_track, reconstructed_track, self.uuid)
        self.file = DataFile(output_analysis, self.absolute_path + '.pickle')