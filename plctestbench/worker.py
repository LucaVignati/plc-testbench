from plctestbench.settings import GlobalSettings

class Worker(object):
    def __init__(self, settings: GlobalSettings) -> None:
        self.settings = settings

    def run(self) -> None:
        pass