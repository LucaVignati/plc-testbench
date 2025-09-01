import datetime

from anytree import LevelOrderIter, search

from .database_manager import DatabaseManager, database_manager_factory
from .models import Run, RunStatus, TestbenchConfiguration, User
from .node import (
    LostSamplesMaskNode,
    Node,
    OriginalTrackNode,
    OutputAnalysisNode,
    ReconstructedTrackNode,
)
from .path_manager import PathManager
from .settings import Settings
from .utils import compute_hash, get_class, progress_monitor


class DataManager(object):

    def __init__(
        self, testbench_settings: TestbenchConfiguration, user: User = None
    ) -> None:
        """
        This class manages the data flow in and out of the data tree.

            Inputs:
                path_manager:   a reference to the path manager is stored
                                and used to set and retrieved file and
                                folder paths.
        """
        self.user = user if user is not None else User.get_default_user()
        root_folder = (
            testbench_settings.root_folder
            if "root_folder" in testbench_settings.__dict__.keys()
            else None
        )
        self.progress_monitor = (
            testbench_settings.progress_monitor
            if "progress_monitor" in testbench_settings.__dict__.keys()
            else progress_monitor
        )
        self.path_manager = PathManager(root_folder)
        self.database_manager: DatabaseManager = database_manager_factory.create(
            testbench_settings.db_platform,
            user=self.user,
            **testbench_settings.__dict__
        )
        self.root_nodes = []
        self.worker_classes = []
        self.node_classes = [
            OriginalTrackNode,
            LostSamplesMaskNode,
            ReconstructedTrackNode,
            OutputAnalysisNode,
        ]
        if not self.database_manager.initialized:
            for node_class, i in zip(
                self.node_classes, range(len(self.node_classes) - 1)
            ):
                self.database_manager.add_node(
                    {"child_collection": self.node_classes[i + 1].__name__},
                    node_class.__name__,
                )

    def run_testbench(self) -> None:
        """
        Run the testbench.
        """
        self._set_run_status(RunStatus.RUNNNING)
        try:
            for root_node in self.progress_monitor(self)(
                self.root_nodes, desc="Audio Tracks"
            ):
                for node in LevelOrderIter(root_node):
                    node.run()
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
            return
        finally:
            self._set_run_status(RunStatus.FAILED)
        self._set_run_status(RunStatus.COMPLETED)

    def set_workers(
        self,
        original_audio_tracks: list,
        packet_loss_simulators: list,
        plc_algorithms: list,
        output_analysers: list,
    ) -> None:
        """
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
        """
        self.worker_classes.append(original_audio_tracks)
        self.worker_classes.append(packet_loss_simulators)
        self.worker_classes.append(plc_algorithms)
        self.worker_classes.append(output_analysers)

    def get_data_trees(self) -> list:
        """
        This function returns a list containing the root nodes of all the trees
        that have been instantiated.
        """
        return self.root_nodes

    def initialize_tree(self) -> str:
        """
        This function is used to initialize the data tree.
        """
        self._recursive_tree_init()
        self._save_run_to_database()
        return self.run._id

    def _recursive_tree_init(self, parent: Node = None, idx: int = 0) -> None:
        """
        This function recursively instanciates all the nodes in the tree.

            Inputs:
                parent:         the newly created node will be attached to the
                                tree as a child of this node.
                                callbacks for each level of the tree.
                idx:            this index is used to move forward and stop the
                                recursion and access the appropriate element of
                                the worker_classes list.
        """
        database = None
        if idx == len(self.worker_classes):
            return
        worker_class = self.worker_classes[idx]
        node_class = self.node_classes[idx]
        for worker, settings in worker_class:

            settings.set_progress_monitor(self.progress_monitor)
            folder_name, absolute_path = self.path_manager.get_node_paths(
                worker, settings, parent
            )
            if parent is None:
                database = self.database_manager
            child = node_class(
                worker=worker,
                settings=settings,
                parent=parent,
                database=database,
                folder_name=folder_name,
                absolute_path=absolute_path,
            )
            if parent is None:
                self.root_nodes.append(child)
            self._recursive_tree_init(child, idx + 1)

    def _save_run_to_database(self) -> None:
        """
        This function is used to save the run as a document in the database.
        """
        self.run: Run = Run()
        run_id: str = ""
        self.run.workers = []
        for worker_class in self.worker_classes:
            workers = []
            for worker, settings in worker_class:
                workers.append(
                    {"name": worker.__name__, "settings": settings.to_dict()}
                )
            self.run.workers.append(workers)

        self.run.nodes = []
        for root_node in self.root_nodes:
            self.run.nodes.extend(
                [{"_id": node.get_id()} for node in list(LevelOrderIter(root_node))]
            )

        for node in self.run.nodes:
            run_id += str(node["_id"])

        self.run._id = str(compute_hash(run_id))
        self.run.creator = self.user.email
        self.run.created_on = datetime.datetime.now()
        self.run.status = RunStatus.CREATED
        self.database_manager.save_run(self.run)

    def load_workers_from_database(self, run_id: int) -> None:
        """
        This function is used to load the workers from the database.
        """
        run: Run = self.database_manager.get_run(run_id)
        self.worker_classes = []
        for worker_type in run.workers:
            workers = []
            for worker in worker_type:
                settings = Settings(worker["settings"])
                settings.__class__ = get_class(worker["name"] + "Settings")
                workers.append((get_class(worker["name"]), settings))
            self.worker_classes.append(workers)

    def _set_run_status(self, state: str) -> None:
        """
        This function is used to set the state of the run in the database.
        """
        self.database_manager.set_run_status(self.run._id, state)

    def get_nodes_by_depth(self, depth: int) -> tuple:
        """
        This function searches all the stored trees and returnes all the nodes at the specified
        depth.

            Inputs:
                depth:  the depth of the desired nodes.
        """
        same_depth_nodes = ()
        for tree in self.root_nodes:
            result = search.findall(
                tree, filter_=lambda node: node.depth == depth, maxlevel=4
            )
            same_depth_nodes += result

        return same_depth_nodes

    def get_leaf_nodes(self) -> tuple:
        """
        This function is a wrapper for the get_nodes_by_depth function.
        It returns the nodes at level 4, which are leaf nodes.
        """
        return self.get_nodes_by_depth(3)
