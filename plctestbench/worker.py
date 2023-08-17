from time import sleep

from plctestbench.settings import Settings
from plctestbench.utils import progress_monitor

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
    def get_node_id(self) -> str:
        return str(hash(self.settings))

    def __str__(self) -> str:
        return self.__class__.__name__

class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def run(self) -> None:
        for _ in progress_monitor(self)(range(1, 10), desc=str(self)):
            sleep(0.1)
