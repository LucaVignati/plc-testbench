from ecctestbench.path_manager import PathManager
from anytree import LevelOrderIter
from tqdm.notebook import tqdm

from ecctestbench.settings import Settings
from .data_manager import DataManager
from .plot_manager import PlotManager


class ECCTestbench(object):
    '''
    The testbench class for the alohaecc algorithm.
    '''

    def __init__(self, packet_loss_simulators: list,
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
        for packet_loss_simulator in packet_loss_simulators:
            simulator = packet_loss_simulator[0]
            loss_model = packet_loss_simulator[1]
            self.packet_drop_simulators.append(simulator(loss_model(settings), settings))
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
        for data_tree in tqdm(data_trees, desc="Audio Tracks"):
            for node in LevelOrderIter(data_tree):
                node.run()

    def plot(self, show=True, to_file=False, original_tracks=False, lost_samples_masks=False, ecc_tracks=False, output_analyses=False, group=False, peaq_summary=False) -> None:
        '''
        Plot all the results
        '''
        if original_tracks:
            plot_manager = PlotManager(self.settings)
            original_track_nodes = self.data_manager.get_nodes_by_depth(0)
            for original_audio_node in original_track_nodes:
                plot_manager.plot_audio_track(original_audio_node, to_file)
        
        if lost_samples_masks:
            plot_manager = PlotManager(self.settings)
            lost_samples_mask_nodes = self.data_manager.get_nodes_by_depth(1)
            for lost_samples_mask_node in lost_samples_mask_nodes:
                plot_manager.plot_lost_samples_mask(lost_samples_mask_node, to_file)

        if ecc_tracks:
            plot_manager = PlotManager(self.settings)
            ecc_track_nodes = self.data_manager.get_nodes_by_depth(2)
            for ecc_track_node in ecc_track_nodes:
                plot_manager.plot_audio_track(ecc_track_node, to_file)

        if output_analyses:
            plot_manager = PlotManager(self.settings)
            output_analysis_nodes = self.data_manager.get_nodes_by_depth(3)
            for output_analysis_node in output_analysis_nodes:
                plot_manager.plot_output_analysis(output_analysis_node, to_file)

        if group:
            plot_manager = PlotManager(self.settings)
            leaf_nodes = self.data_manager.get_leaf_nodes()
            for leaf_node in leaf_nodes:
                ancestors = leaf_node.ancestors
                plot_manager.plot_audio_track(ancestors[0], to_file)
                plot_manager.plot_lost_samples_mask(ancestors[1], to_file)
                plot_manager.plot_audio_track(ancestors[2], to_file)
                plot_manager.plot_output_analysis(leaf_node, to_file)

        if peaq_summary:
            plot_manager = PlotManager(self.settings)
            output_analysis_nodes = self.data_manager.get_nodes_by_depth(3)
            plot_manager.plot_peaq_summary(output_analysis_nodes, to_file)

        if show:
            PlotManager.show()
        