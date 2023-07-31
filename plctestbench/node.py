from anytree import NodeMixin
import numpy as np
from copy import copy

from plctestbench.worker import Worker
from plctestbench.file_wrapper import FileWrapper, AudioFile, DataFile
from plctestbench.settings import Settings

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    def __init__(self, file: FileWrapper=None, worker: Worker=None, settings: Settings=None, absolute_path: str=None, parent=None, database=None, folder_name=None) -> None:
        self.file = file
        self.settings = copy(settings)
        if parent!=None:
            self.settings.inherit_from(parent.settings)
        self.worker = worker(self.settings) if worker!=None else None
        self.database = database
        self.parent = parent
        self.folder_name = folder_name
        self.absolute_path = absolute_path

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
    
    def _get_database(self):
        return self.root.database

    def _load_from_database(self) -> dict:
        return self._get_database().find_node(self.get_id(), type(self).__name__)

    def _save_to_database(self):
        entry = self.settings.get_all().copy()
        entry["filepath"] = self.file.get_path()
        entry["_id"] = self.get_id()
        entry["file_hash"] = str(hash(self.file))
        entry["parent"] = self.parent.get_id() if self.parent!=None else None
        self._get_database().add_node(entry, type(self).__name__)

    def get_id(self) -> str:
        return str(hash(self.settings))
    
    def run(self):
        if self.worker != None:
            self.worker.set_uuid(self.get_id())

        # Load from database if possible, otherwise run the worker
        current_node = self._load_from_database()
        if current_node == None:
            self._run()
            self._save_to_database()
        else:
            self.file = FileWrapper.from_path(current_node["filepath"])

            # Manage consistency between database and filesystem
            if str(hash(self.file)) != current_node["file_hash"]:
                if self.parent == None:
                    raise Exception("The following audio file has changed: " + self.file.get_path())
                else:
                    self._get_database().delete_node(self.get_id())
                    self.run()
    
    def __str__(self) -> str:
        return "file: " + str(self.file) + '\n' +\
               "worker: " + str(self.worker) + '\n' +\
               "folder name: " + str(self.folder_name) + '\n' + \
               "absolute path: " + str(self.absolute_path)

class OriginalTrackNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None, database=None, folder_name=None) -> None:
        super().__init__(file=file, worker=worker, settings=settings, absolute_path=absolute_path, parent=parent, database=database, folder_name=folder_name)
        self.file = AudioFile(path=self.absolute_path + '.wav')
        self.settings.add('fs', self.file.get_samplerate())

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_track_name(self) -> str:
        return self.absolute_path.rpartition("/")[2].split(".")[0]
    
    def load(self) -> None:
        self.file = AudioFile.from_path(self.absolute_path + '.wav')
        self.file.persist = False
        self.file.load()

    def _run(self) -> None:
        print(self.get_track_name())
        self.get_worker().run()
    
class LostSamplesMaskNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None, database=None, folder_name=None) -> None:
        super().__init__(file=file, worker=worker, settings=settings, absolute_path=absolute_path, parent=parent, database=database, folder_name=folder_name)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root
    
    def load(self) -> None:
        self.file = DataFile(path=self.absolute_path + '.npy', persist=False)
        #self.file.load()
    
    def _run(self) -> None:
        original_track_data = self.get_original_track().get_data()
        num_samples = len(original_track_data)
        lost_samples_idx = self.get_worker().run(num_samples)
        self.file = DataFile(lost_samples_idx, self.absolute_path + '.npy')

class ReconstructedTrackNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None, database=None, folder_name=None) -> None:
        super().__init__(file=file, worker=worker, settings=settings, absolute_path=absolute_path, parent=parent, database=database, folder_name=folder_name)

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
    
    def _run(self) -> None:
        original_track = self.get_original_track()
        original_track_data = original_track.get_data()
        lost_samples_idx = self.get_lost_samples_mask().get_data()
        reconstructed_track = self.get_worker().run(original_track_data, lost_samples_idx)
        self.file = AudioFile.from_audio_file(original_track, reconstructed_track, self.absolute_path + '.wav')

class OutputAnalysisNode(Node):
    def __init__(self, file=None, worker=None, settings=None, absolute_path=None, parent=None, database=None, folder_name=None) -> None:
        super().__init__(file=file, worker=worker, settings=settings, absolute_path=absolute_path, parent=parent, database=database, folder_name=folder_name)

    def get_data(self) -> np.ndarray:
        return self.file.get_data()

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root
    
    def get_lost_samples_mask_node(self) -> LostSamplesMaskNode:
        return self.ancestors[1]

    def get_reconstructed_track_node(self) -> ReconstructedTrackNode:
        return self.ancestors[2]
    
    def load(self) -> None:
        self.file = DataFile(path=self.absolute_path + '.pickle', persist=False)
        #self.file.load()

    def _run(self) -> None:
        original_track = self.get_original_track()
        reconstructed_track = self.get_reconstructed_track()
        output_analysis = self.get_worker().run(original_track, reconstructed_track)
        self.file = DataFile(output_analysis, self.absolute_path + '.pickle')
