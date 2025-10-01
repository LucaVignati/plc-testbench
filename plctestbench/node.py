from copy import deepcopy
from anytree import NodeMixin
import numpy as np

from plctestbench.worker import Worker
from plctestbench.file_wrapper import FileWrapper, AudioFile, DataFile
from plctestbench.settings import Settings
from plctestbench.utils import dummy_progress_bar

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    def __init__(self, file: FileWrapper | None = None,
                       worker: Worker | None = None,
                       settings: Settings | None = None,
                       absolute_path: str | None = None,
                       parent = None,
                       database = None,
                       folder_name = None) -> None:
        self.file = file
        self.settings = deepcopy(settings)
        if parent is not None:
            self.settings.inherit_from(parent.settings) #type: ignore
        self.worker = worker(self.settings) if worker is not None else None #type: ignore
        self.database = database
        self.parent = parent
        self.folder_name = folder_name
        self.absolute_path = absolute_path
        self.persistent = True

    def set_folder_name(self, folder_name) -> None:
        self.folder_name = folder_name

    def set_file(self, file) -> None:
        self.file = file

    def get_file(self) -> FileWrapper | None:
        return self.file

    def get_folder_name(self) -> str | None:
        return self.folder_name

    def set_path(self, absolute_path) -> None:
        self.absolute_path = absolute_path

    def get_path(self) -> str | None:
        return self.absolute_path

    def get_worker(self) -> Worker | None:
        return self.worker

    def get_original_track(self) -> AudioFile | FileWrapper | None:
        return self.root.get_file()

    def get_lost_samples_mask(self) -> DataFile:
        return self.ancestors[1].get_file()

    def get_reconstructed_track(self) -> AudioFile:
        return self.ancestors[2].get_file()

    def get_setting(self, setting_name: str):
        return self.settings.get(setting_name) #type: ignore

    def _get_database(self):
        return self.root.database

    def _load_from_database(self) -> dict:
        return self._get_database().find_node(self.get_id(), type(self).__name__) #type: ignore

    def _save_to_database(self):
        entry = self.settings.to_dict().copy() #type: ignore
        entry["filepath"] = self.file.get_path() #type: ignore
        entry["_id"] = self.get_id()
        entry["file_hash"] = str(hash(self.file))
        entry["parent"] = self.parent.get_id() if self.parent is not None else None
        entry["persistent"] = self.persistent
        self._get_database().add_node(entry, type(self).__name__) #type: ignore

    def get_id(self) -> str:
        return str(hash(self.settings))

    def _run(self):
        pass

    def run(self):

        # Load from database if possible, otherwise run the worker
        current_node = self._load_from_database()
        if current_node:
            self.persistent = current_node["persistent"]
        if self.persistent is False or current_node is None:
            self._run()
            self._save_to_database()
        else:
            self.file = FileWrapper.from_path(current_node["filepath"])

            # Manage consistency between database and filesystem
            if str(hash(self.file)) != current_node["file_hash"]:
                if self.parent is None:
                    raise Exception("The following audio file has changed: " + self.file.get_path()) #type: ignore
                else:
                    self._get_database().delete_node(self.get_id()) #type: ignore
                    self.run()
            else:
                # Dummy progress bar needed when not running the worker
                dummy_progress_bar(self.worker)
    
    def __str__(self) -> str:
        return "file: " + str(self.file) + '\n' +\
               "worker: " + str(self.worker) + '\n' +\
               "folder name: " + str(self.folder_name) + '\n' + \
               "absolute path: " + str(self.absolute_path)

class OriginalTrackNode(Node):
    def __init__(self,
                 file=None,
                 worker=None,
                 settings=None,
                 absolute_path=None,
                 parent=None,
                 database=None,
                 folder_name=None) -> None:
        super().__init__(file=file,
                         worker=worker,
                         settings=settings,
                         absolute_path=absolute_path,
                         parent=parent,
                         database=database,
                         folder_name=folder_name)
        self.file = AudioFile(path=self.absolute_path + '.wav') #type: ignore
        self.settings.add('fs', self.file.get_samplerate()) #type: ignore

    def get_data(self) -> np.ndarray:
        return self.file.get_data() #type: ignore

    def get_track_name(self) -> str:
        return self.absolute_path.rpartition("/")[2].split(".")[0] #type: ignore

    def _run(self) -> None:
        print(self.get_track_name())
        self.get_worker().run() #type: ignore
        self.persistent = self.get_worker().is_persistent() #type: ignore

class LostSamplesMaskNode(Node):
    def __init__(self, file = None,
                 worker = None,
                 settings = None,
                 absolute_path = None,
                 parent = None,
                 database = None,
                 folder_name = None) -> None:
        super().__init__(file = file,
                         worker = worker,
                         settings = settings,
                         absolute_path = absolute_path,
                         parent = parent,
                         database = database,
                         folder_name = folder_name)

    def get_data(self) -> np.ndarray:
        return self.file.get_data() #type: ignore

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root #type: ignore

    def _run(self) -> None:
        original_track_data = self.get_original_track().get_data() #type: ignore
        num_samples = len(original_track_data) #type: ignore
        lost_samples_idx = self.get_worker().run(num_samples) #type: ignore
        self.persistent = self.get_worker().is_persistent() #type: ignore
        self.file = DataFile(lost_samples_idx, self.absolute_path + '.npy') #type: ignore

class ReconstructedTrackNode(Node):
    def __init__(self, file = None,
                 worker = None,
                 settings = None,
                 absolute_path = None,
                 parent = None, database = None,
                 folder_name = None) -> None:
        super().__init__(file = file,
                         worker = worker,
                         settings = settings,
                         absolute_path = absolute_path,
                         parent = parent,
                         database = database,
                         folder_name = folder_name)

    def get_data(self) -> np.ndarray:
        return self.file.get_data() #type: ignore

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root #type: ignore

    def get_lost_samples_mask_node(self) -> LostSamplesMaskNode:
        return self.ancestors[1]

    def _run(self) -> None:
        original_track = self.get_original_track()
        original_track_data = original_track.get_data() #type: ignore
        lost_samples_idx = self.get_lost_samples_mask().get_data()
        reconstructed_track = self.get_worker().run(original_track_data, lost_samples_idx) #type: ignore
        self.persistent = self.get_worker().is_persistent() #type: ignore
        self.file = AudioFile.from_audio_file(original_track, reconstructed_track, self.absolute_path + '.wav') #type: ignore

class OutputAnalysisNode(Node):
    def __init__(self,
                 file = None,
                 worker = None,
                 settings = None,
                 absolute_path = None,
                 parent = None,
                 database = None,
                 folder_name = None) -> None:
        super().__init__(file = file,
                         worker = worker,
                         settings = settings,
                         absolute_path = absolute_path,
                         parent = parent,
                         database = database,
                         folder_name = folder_name)

    def get_data(self) -> np.ndarray:
        return self.file.get_data() #type: ignore

    def get_original_track_node(self) -> OriginalTrackNode:
        return self.root #type: ignore

    def get_lost_samples_mask_node(self) -> LostSamplesMaskNode:
        return self.ancestors[1]

    def get_reconstructed_track_node(self) -> ReconstructedTrackNode:
        return self.ancestors[2]

    def _run(self) -> None:
        original_track = self.get_original_track()
        reconstructed_track = self.get_reconstructed_track()
        lost_samples_idx = self.get_lost_samples_mask()
        output_analysis = self.get_worker().run(original_track, reconstructed_track, lost_samples_idx) #type: ignore
        self.persistent = self.get_worker().is_persistent() #type: ignore
        self.file = DataFile(output_analysis, self.absolute_path + '.pickle') #type: ignore
