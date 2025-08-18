from .data_manager import DataManager
from .plot_manager import PlotManager

class PLCTestbench(object):
    '''
    This class is the main class of the testbench. It is responsible for
    initialising the testing components and running the testbench.
    '''

    def __init__(self, original_audio_tracks: list | None = None,
                 packet_loss_simulators: list | None = None,
                 plc_algorithms: list | None = None,
                 output_analysers: list | None = None,
                 testbench_settings: dict | None = None,
                 user: dict | None = None,
                 run_id: int | None = None):
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
        if testbench_settings is None:
            raise ValueError("testbench_settings must be provided")
        self.data_manager = DataManager(testbench_settings, user)

        if run_id:
            self.data_manager.load_workers_from_database(run_id)
        elif packet_loss_simulators is None \
             or plc_algorithms is None \
             or output_analysers is None:
            raise ValueError("packet_loss_simulators, \
                              plc_algorithms and output_analysers \
                              must be provided if no run_id is provided")
        else:
            self.data_manager.set_workers(original_audio_tracks,
                                          packet_loss_simulators,
                                          plc_algorithms,
                                          output_analysers)

        self.run_id = self.data_manager.initialize_tree()

    def run(self) -> None:
        '''
        Run the testbench.
        '''
        self.data_manager.run_testbench()
        print("testbench.run finished!")

    def plot(self, plot_settings={}, show=True, to_file=False, original_tracks=False, lost_samples_masks=False, reconstructed_tracks=False, output_analyses=False, group=False, peaq_summary=False) -> None:
        '''
        Plot all the results

            Inputs:
                show:                   shows plots in Jupyter Notebook
                to_file:                plots will be saved as files
                original_tracks:        plots audio track in Jupyter Notebook
                lost_samples_masks:     plots a diagram of the lost samples in Jupyter Notebook
                reconstructed_tracks:   plots audio signals after reconstruction in Jupyter Notebook
                output_analyses:        plots the results of the metrics in Jupyter Notebook
                group:                  combined setting for the 4 parameters above
                peaq_summary:           plots the peaq results (currently not usable, implementation incomplete)
        '''
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
        print("testbench.plot finished!")
        