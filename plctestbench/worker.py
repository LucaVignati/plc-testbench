from plctestbench.settings import Settings
from plctestbench.file_wrapper import AudioFile

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> None:
        pass

class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)