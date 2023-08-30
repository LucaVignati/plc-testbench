from time import sleep

from plctestbench.settings import Settings

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.progress_monitor = settings.get_progress_monitor()(self)
        
    def get_node_id(self) -> str:
        return str(hash(self.settings))

    def __str__(self) -> str:
        return self.__class__.__name__

class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def run(self) -> None:
        for _ in self.progress_monitor(range(1), desc=str(self)):
            sleep(0.1)
