from plctestbench.settings import Settings

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def __str__(self) -> str:
        return self.__class__.__name__

class OriginalAudio(Worker):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
