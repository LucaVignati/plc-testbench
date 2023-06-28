import typing
import anytree.search as search
from anytree import LevelOrderIter
from plctestbench.path_manager import PathManager
from .database_manager import DatabaseManager
from .node import ReconstructedTrackNode, LostSamplesMaskNode, Node, OriginalTrackNode, OutputAnalysisNode
from .file_wrapper import AudioFile, DataFile
from .settings import Settings, OriginalAudioSettings
from .utils import *

def recursive_tree_init(parent: Node, worker_classes: list, node_classes: list, idx: int):
    '''
    This function recursively instanciates all the nodes in the tree.

        Inputs:
            parent:         the newly created node will be attached to the
                            tree as a child of this node.
            worker_classes: this list contains the workers and associated
                            callbacks for each level of the tree.
            idx:            this index is used to move forward and stop the
                            recursion and access the appropriate element of
                            the worker_classes list.
    '''
    if idx == len(worker_classes):
        return
    worker_class = worker_classes[idx]
    node_class = node_classes[idx + 1]
    for worker, settings in worker_class:
        child = node_class(worker=worker, settings=settings, parent=parent)
        PathManager.set_node_paths(child)
        recursive_tree_init(child, worker_classes, node_classes, idx + 1)

class DataManager(object):

    def __init__(self, testbench_settings: dict, user: dict) -> None:
        '''
        This class manages the data flow in and out of the data tree.

            Inputs:
                path_manager:   a reference to the path manager is stored
                                and used to set and retrieved file and
                                folder paths.
        '''
        self.user = user if user is not None else {'email': 'default', 'first_name': 'Mario', 'last_name': 'Rossi', 'locale': 'it_IT', 'image_link': ''}
        root_folder = testbench_settings['root_folder'] if 'root_folder' in testbench_settings.keys() else '../original_tracks'
        db_ip = testbench_settings['db_ip'] if 'db_ip' in testbench_settings.keys() else 'localhost'
        db_port = int(testbench_settings['db_port']) if 'db_port' in testbench_settings.keys() else 27017
        db_username = testbench_settings['db_username'] if 'db_username' in testbench_settings.keys() else 'admin'
        db_password = testbench_settings['db_password'] if 'db_password' in testbench_settings.keys() else 'admin'
        
        self.path_manager = PathManager(root_folder)
        self.database_manager = DatabaseManager(ip=db_ip, port=db_port, username=db_username, password=db_password, user=self.user)
        self.root_nodes = []
        self.worker_classes = []
        self.node_classes = [
            OriginalTrackNode,
            LostSamplesMaskNode,
            ReconstructedTrackNode,
            OutputAnalysisNode
        ]
        if not self.database_manager.initialized:
            for node_class, i in zip(self.node_classes, range(len(self.node_classes) - 1)):
                self.database_manager.add_node({"child_collection": self.node_classes[i + 1].__name__}, node_class.__name__)

    def set_workers(self, packet_loss_simulators: list(),
                          plc_algorithms: list(),
                          output_analysers: list()) -> None:
        '''
        This function stores into the DataManager class the instances of the workers
        to be used during the simulations, associated with the callback that manages
        the execution of the relative worker.

            Inputs:
                packet_loss_simulators: this list must contain one or more instances
                                        of a subclass of the PacketLossSimulator
                                        class
                plc_algorithms:         this list must contain one or more instances
                                        of a subclass of the PLCAlgorithm
                                        class
                output_analysers:       this list must contain one or more instances
                                        of a subclass of the OutputAnalyser
                                        class
        '''
        self.worker_classes.append(packet_loss_simulators)
        self.worker_classes.append(plc_algorithms)
        self.worker_classes.append(output_analysers)

    def get_data_trees(self) -> list:
        '''
        This function returns a list containing the root nodes of all the trees
        that have been instantiated.
        '''
        return self.root_nodes

    def initialize_tree(self) -> None:
        '''
        This function instanciates each node of the tree where the workers and their results
        will be stored. This function works in a recursive fashion.

            Inputs:
                track_path: a string representing the absolute path to an original audio
                            track.
        '''
        for track_path in self.path_manager.get_original_tracks():
            track = AudioFile(path=track_path)
            root_node = OriginalTrackNode(file=track, settings=OriginalAudioSettings(hash(track)), database=self.database_manager)
            PathManager.set_root_node_path(root_node)
            self.root_nodes.append(root_node)
            recursive_tree_init(root_node, self.worker_classes, self.node_classes, 0)
        self._save_run_to_database()

    def _save_run_to_database(self):
        '''
        This function is used to save the run as a document in the database.
        '''
        run = {}
        run_id = ''
        run['workers'] = []
        for worker_class in self.worker_classes:
            workers = []
            for worker, settings in worker_class:
                workers.append({"name": worker.__name__, "settings": settings.get_all()})
                run_id += str(hash(settings))
            run['workers'].append(workers)

        run['nodes'] = []
        for root_node in self.root_nodes:
            run['nodes'].extend([{"_id": node.get_id()} for node in list(LevelOrderIter(root_node))])

        run['_id'] = compute_hash(run_id)
        self.database_manager.save_run(run)

    def load_workers_from_database(self, run_id: int):
        '''
        This function is used to load the workers from the database.
        '''
        run = self.database_manager.get_run(run_id)
        self.worker_classes = []
        for worker_type in run['workers']:
            workers = []
            for worker in worker_type:
                workers.append((get_class(worker['name']), Settings(worker['settings'])))
            self.worker_classes.append(workers)

    def get_nodes_by_depth(self, depth: int) -> typing.Tuple:
        '''
        This function searches all the stored trees and returnes all the nodes at the specified
        depth.

            Inputs:
                depth:  the depth of the desired nodes.
        '''
        same_depth_nodes = ()
        for tree in self.root_nodes:
            result = search.findall(tree, filter_=lambda node: node.depth==depth, maxlevel=4)
            same_depth_nodes += result
        
        return same_depth_nodes

    def get_leaf_nodes(self) -> typing.Tuple:
        '''
        This function is a wrapper for the get_nodes_by_depth function.
        It returns the nodes at level 4, which are leaf nodes.
        '''
        return self.get_nodes_by_depth(3)