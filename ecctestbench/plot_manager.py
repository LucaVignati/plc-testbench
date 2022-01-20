import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ecctestbench.settings import Settings
from .node import ECCTrackNode, Node, OriginalTrackNode, LostSamplesMaskNode, OutputAnalysisNode
from .output_analyser import MSECalculator, SpectralEnergyCalculator, PEAQCalculator

class PlotManager(object):

    def __init__(self, settings: Settings, rows:int = None, cols:int = None) -> None:
        '''
        Base class for plotting results

        '''
        self.dpi = settings.dpi
        self.linewidth = settings.linewidth
        self.figsize = settings.figsize
        mpl.rcParams['agg.path.chunksize'] = 10000

    def plot_audio_track(self, node: Node, to_file=False) -> None:
        '''
        Plot the original input file
        '''
        audio_file = node.get_file()
        samplerate = audio_file.get_samplerate()
        n_channels = audio_file.get_channels()
        audio_file_data = audio_file.get_data()
        x = np.arange(0, len(audio_file_data)/samplerate, 1/samplerate)

        fig, ax = plt.subplots(n_channels, 1, sharex=True, figsize=self.figsize, dpi=self.dpi)
        if issubclass(node.__class__, OriginalTrackNode):
            fig.suptitle("Original Track")
        elif issubclass(node.__class__, ECCTrackNode):
            fig.suptitle("ECC Track")

        for n in range(n_channels):
            if audio_file_data.ndim > 1:
                audio_channel_data = audio_file_data[:, n]
            else:
                audio_channel_data = audio_file_data
                ax = [ax]
            #ax = fig.add_axes([0, 0, 1, 1])
            ax[n].plot(x, audio_channel_data, linewidth=self.linewidth)
            ax[n].set_title("Channel " + str(n + 1) + "/" + str(n_channels))
            ax[n].set_xlabel("Time [s]")
            ax[n].set_ylabel("Normalized Amplitude")
            ax[n].set_xlim(0, x[-1])
            ax[n].set_ylim(-1, 1)
            if to_file:
                fig.savefig(node.get_path(), bbox_inches='tight')
    
    def plot_lost_samples_mask(self, node: LostSamplesMaskNode, to_file=False) -> None:
        '''
        Plot the lost samples mask data
        '''
        lost_samples_idx = node.get_file().get_data()
        original_track = node.get_original_track()
        samplerate = original_track.get_samplerate()
        original_track_length = (len(original_track.get_data()) - 1)/samplerate
        lost_samples_times = lost_samples_idx / samplerate
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.suptitle("Lost Samples")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.vlines(lost_samples_times, 0, 1, linewidth=self.linewidth)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lost Samples")
        ax.set_xlim(0, original_track_length)
        if to_file:
            fig.savefig(node.get_path(), bbox_inches='tight')

    def plot_output_analysis(self, node: OutputAnalysisNode, to_file=False) -> None:
        '''
        Plot the output analysis data
        '''
        original_track = node.get_original_track()
        samplerate = original_track.get_samplerate()
        n_channels = original_track.get_channels()
        original_track_length = (len(original_track.get_data()) - 1)/samplerate
        data = node.get_file().get_data()
        x = np.arange(0, original_track_length, original_track_length/len(data))
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        if issubclass(node.worker.__class__, MSECalculator):
            name = "Mean Square Error"
            fig.suptitle(name)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(name)
            ax.set_xlim(0, original_track_length)
            for n in range(n_channels):
                if data.ndim > 1:
                    channel_data = data[:, n]
                else:
                    channel_data = data
                label = "Channel " + str(n + 1)
                ax.plot(x, channel_data, label=label)
            plt.legend(loc="upper left")
            if to_file:
                fig.savefig(node.get_path(), bbox_inches='tight')
        if issubclass(node.worker.__class__, SpectralEnergyCalculator):
            name = "Spectral Energy"

    def show() -> None:
        plt.show()