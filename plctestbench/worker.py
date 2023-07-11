from time import sleep
from tqdm.auto import tqdm as std_tqdm

from plctestbench.settings import Settings
from plctestbench.file_wrapper import AudioFile

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.progress_monitor = lambda caller: std_tqdm
        self.uuid = None

    def set_progress_monitor(self, progress_monitor):
        self.progress_monitor = progress_monitor
        
    def set_uuid(self, uuid: str):
        self.uuid = uuid

    def run(self) -> None:
        pass
    
class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def run(self) -> None:
        for idx in self.progress_monitor(self)(range(1, 10), desc=self.__str__()):
            sleep(0.1)
            
    def __str__(self) -> str:
        return __class__.__name__