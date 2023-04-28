from plctestbench.path_manager import PathManager
from anytree import LevelOrderIter
from tqdm.notebook import tqdm

from plctestbench.settings import GlobalSettings, PlotsSettings
from .data_manager import DataManager
from .plot_manager import PlotManager

import uuid as Uuid

class PLCTestbench(object):
    '''
    This class is the main class of the testbench. It is responsible for
    initialising the testing components and running the testbench.
    '''

    def __init__(self, packet_loss_simulators: list,
                 plc_algorithms: list,
                 output_analysers: list,
                 global_settings_list: list,
                 data_manager: DataManager,
                 path_manager: PathManager,
                 uuid: str = None):
        '''
        Initialise the parameters and testing components.

            Input:
                plc_algorithm: The chosen method of error concealment of the
                input signal after packet loss is applied
                buffer_size: Default buffer size. Can be overriden if
                necessary.
                fs: Sample Rate. Argument can be overriden if necessary.
                chans: Number of Channels
        '''
        self.global_settings_list = global_settings_list
        self.data_manager = data_manager
        self.path_manager = path_manager

        self.data_manager.set_workers(packet_loss_simulators,
                                      plc_algorithms,
                                      output_analysers)
        
        self.uuid = uuid if uuid != None else str(Uuid.uuid4())

        for trackpath in self.path_manager.get_original_tracks():
            self.data_manager.initialize_tree(trackpath, global_settings_list)

    def run(self) -> None:
        '''
        Run the testbench.
        '''
        data_trees = self.data_manager.get_data_trees()
        progress = self.global_settings_list[0].__progress_monitor__(self)(data_trees, desc="Audio Tracks", mininterval=0.1, maxinterval=0.1)
        for data_tree in progress:
            for node in LevelOrderIter(data_tree):
                node.run()

    def plot(self, show=True, to_file=False, original_tracks=False, lost_samples_masks=False, reconstructed_tracks=False, output_analyses=False, group=False, peaq_summary=False) -> None:
        '''
        Plot all the results
        '''
        plot_settings = PlotsSettings()
        plot_settings.inherit_from(self.global_settings_list[0])
        if original_tracks:
            plot_manager = PlotManager(plot_settings)
            original_track_nodes = self.data_manager.get_nodes_by_depth(0)
            for original_audio_node in original_track_nodes:
                plot_manager.plot_audio_track(original_audio_node, to_file)
        
        if lost_samples_masks:
            plot_manager = PlotManager(plot_settings)
            lost_samples_mask_nodes = self.data_manager.get_nodes_by_depth(1)
            for lost_samples_mask_node in lost_samples_mask_nodes:
                plot_manager.plot_lost_samples_mask(lost_samples_mask_node, to_file)

        if reconstructed_tracks:
            plot_manager = PlotManager(plot_settings)
            reconstructed_track_nodes = self.data_manager.get_nodes_by_depth(2)
            for reconstructed_track_node in reconstructed_track_nodes:
                plot_manager.plot_audio_track(reconstructed_track_node, to_file)

        if output_analyses:
            plot_manager = PlotManager(plot_settings)
            output_analysis_nodes = self.data_manager.get_nodes_by_depth(3)
            for output_analysis_node in output_analysis_nodes:
                plot_manager.plot_output_analysis(output_analysis_node, to_file)

        if group:
            plot_manager = PlotManager(plot_settings)
            leaf_nodes = self.data_manager.get_leaf_nodes()
            for leaf_node in leaf_nodes:
                ancestors = leaf_node.ancestors
                plot_manager.plot_audio_track(ancestors[0], to_file)
                plot_manager.plot_lost_samples_mask(ancestors[1], to_file)
                plot_manager.plot_audio_track(ancestors[2], to_file)
                plot_manager.plot_output_analysis(leaf_node, to_file)

        if peaq_summary:
            plot_manager = PlotManager(plot_settings)
            output_analysis_nodes = self.data_manager.get_nodes_by_depth(3)
            plot_manager.plot_peaq_summary(output_analysis_nodes, to_file)

        if show:
            PlotManager.show()
        