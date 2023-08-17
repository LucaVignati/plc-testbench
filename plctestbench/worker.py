from time import sleep

from plctestbench.settings import Settings
from plctestbench.file_wrapper import AudioFile
from plctestbench.utils import progress_monitor

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
    def get_node_id(self) -> str:
        return str(hash(self.settings))

    def run(self) -> None:
        pass
    
class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def run(self) -> None:
        for _ in progress_monitor(range(1, 10), desc=self.__str__()):
            sleep(0.1)
            
    def __str__(self) -> str:
        return __class__.__name__