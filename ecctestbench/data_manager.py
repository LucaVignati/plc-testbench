import typing
import anytree.search as search
from numpy import ndarray
import soundfile as sf
import numpy as np
from ecctestbench.path_manager import PathManager
from .node import ECCTrackNode, LostSamplesMaskNode, Node, OriginalTrackNode, OutputAnalysisNode
from .file_wrapper import AudioFile, DataFile

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
    for worker in worker_class:
        child = node_class(worker=worker, parent=parent)
        PathManager.set_node_paths(child)
        recursive_tree_init(child, worker_classes, node_classes, idx + 1)

class DataManager(object):

    def __init__(self, path_manager: PathManager) -> None:
        '''
        This class manages the data flow in and out of the data tree.

            Inputs:
                path_manager:   a reference to the path manager is stored
                                and used to set and retrieved file and
                                folder paths.
        '''
        self.path_manager = path_manager
        self.root_nodes = list()
        self.worker_classes = list()
        self.node_classes = [
            OriginalTrackNode,
            LostSamplesMaskNode,
            ECCTrackNode,
            OutputAnalysisNode
        ]

    def set_workers(self, packet_loss_simulators: list(),
                          ecc_algorithms: list(),
                          output_analysers: list()) -> None:
        '''
        This function stores into the DataManager class the instances of the workers
        to be used during the simulations, associated with the callback that manages
        the execution of the relative worker.

            Inputs:
                packet_loss_simulators: this list must contain one or more instances
                                        of a subclass of the PacketLossSimulator
                                        class
                ecc_algorithms:         this list must contain one or more instances
                                        of a subclass of the ECCAlgorithm
                                        class
                output_analysers:       this list must contain one or more instances
                                        of a subclass of the OutputAnalyser
                                        class
        '''
        self.worker_classes.append(packet_loss_simulators)
        self.worker_classes.append(ecc_algorithms)
        self.worker_classes.append(output_analysers)

    def get_data_trees(self) -> list:
        '''
        This function returns a list containing the root nodes of all the trees
        that have been instantiated.
        '''
        return self.root_nodes

    def initialize_tree(self, track_path: str) -> None:
        '''
        This function instanciates each node of the tree where the workers and their results
        will be stored. This function works in a recursive fashion.

            Inputs:
                track_path: a string representing the absolute path to an original audio
                            track.
        '''
        track = AudioFile.from_path(track_path)
        root_node = OriginalTrackNode(file=track)
        PathManager.set_root_node_path(root_node)
        self.root_nodes.append(root_node)
        recursive_tree_init(root_node, self.worker_classes, self.node_classes, 0)

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