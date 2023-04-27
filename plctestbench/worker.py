from plctestbench.settings import GlobalSettings

class Worker(object):
    def __init__(self, settings: GlobalSettings) -> None:
        self.settings = settings

    def run(self) -> None:
        pass

class OriginalTrackWorker(Worker):
    def __init__(self, settings: GlobalSettings) -> None:
        super().__init__(settings)

    def run(self, uuid: str) -> None:
        self.uuid = uuid
        for idx in self.settings.__progress_monitor__(self)(range(0, 10), desc=self.__str__(), mininterval=0.1, maxinterval=0.1):
            print("Worker %s (%s)" % (self.uuid, self.__class__.__name__))
            #sleep(1)
