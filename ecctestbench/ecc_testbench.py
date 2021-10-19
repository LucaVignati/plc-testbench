from os import path
from alohaecc.path_manager import PathManager
from anytree import LevelOrderIter

from ecctestbench.settings import Settings
from .data_manager import DataManager
from .ecc_algorithm import ECCAlgorithm, ZerosEcc, LastPacketEcc
from .output_analyser import OutputAnalyser, MSECalculator, PEAQCalculator
from .packet_loss_simulator import (PacketLossSimulator,
                                    BasePacketLossSimulator,
                                    BinomialPacketLossSimulator,
                                    BinomialSampleLossSimulator)


class ECCTestbench(object):
    '''
    The testbench class for the alohaecc algorithm.
    '''

    def __init__(self, packet_drop_simulators: list,
                 ecc_algorithms: list,
                 output_analysers: list,
                 settings: Settings,
                 data_manager: DataManager,
                 path_manager: PathManager):
        '''
        Initialise the parameters and testing components.

            Input:
                ecc_algorithm: The chosen method of error concealment of the
                input signal after packet loss is applied
                buffer_size: Default buffer size. Can be overriden if
                necessary.
                fs: Sample Rate. Argument can be overriden if necessary.
                chans: Number of Channels
        '''
        self.packet_drop_simulators = list()
        self.ecc_algorithms = list()
        self.output_analysers = list()
        for packet_drop_simulator in packet_drop_simulators:
            self.packet_drop_simulators.append(packet_drop_simulator(settings))
        for ecc_algorithm in ecc_algorithms:
            self.ecc_algorithms.append(ecc_algorithm(settings))
        for output_analyser in output_analysers:
            self.output_analysers.append(output_analyser(settings))
        self.settings = settings
        self.data_manager = data_manager
        self.path_manager = path_manager

        self.data_manager.set_workers(self.packet_drop_simulators,
                                      self.ecc_algorithms,
                                      self.output_analysers)

        for trackpath in self.path_manager.get_original_tracks():
            self.data_manager.initialize_tree(trackpath)

    def run(self) -> None:
        '''
        Run the testbench.
        '''
        data_trees = self.data_manager.get_data_trees()
        for data_tree in data_trees:
            for node in LevelOrderIter(data_tree):
                #print(node)
                node.run()
                #print(node)

    def plot(self) -> None:
        '''
        Plot all the results
        '''
        original_audio_nodes = self.data_manager.get_nodes_by_depth(0)
        for original_audio_node in original_audio_nodes:
            original_audio_node.plot(to_file=True)
        
        lost_samples_mask_nodes = self.data_manager.get_nodes_by_depth(1)
        for lost_samples_mask_node in lost_samples_mask_nodes:
            lost_samples_mask_node.plot(to_file=True)
        