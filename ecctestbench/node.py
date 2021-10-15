from anytree import NodeMixin

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    def __init__(self, file=None, worker=None, parent=None) -> None:
        self.file = file
        self.worker = worker
        self.parent = parent
        self.folder_name = None

    def set_folder_name(self, folder_name) -> None:
        self.folder_name = folder_name

    def set_file(self, file) -> None:
        self.file = file

    def run(self) -> None:
        if self.worker is not None:
            self.data = self.worker.run(self)

    def get_file(self):
        self.file.seek(0,0)
        return self.file
    
    def get_folder_name(self):
        return self.folder_name
    
    def get_worker(self):
        return self.worker
    
    def __str__(self) -> str:
        return "file: " + str(self.file) + '\n' +\
               "worker: " + str(self.worker) + '\n' +\
                "callback: " + str(self.callback) + '\n' +\
                "folder name: " + str(self.folder_name)