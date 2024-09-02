from plctestbench.settings import Settings
from plctestbench.utils import dummy_progress_bar

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent = True
        self.progress_monitor = settings.get_progress_monitor()(self)
        
    def set_progress_monitor(self, progress_monitor) -> None:
        self.progress_monitor = progress_monitor

    def get_node_id(self) -> str:
        return str(hash(self.settings))

    def is_persistent(self) -> bool:
        return self.persistent

    def __str__(self) -> str:
        return self.__class__.__name__

class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def run(self) -> None:
        dummy_progress_bar(self)
