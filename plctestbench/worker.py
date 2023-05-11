from plctestbench.settings import Settings

class Worker(object):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> None:
        pass