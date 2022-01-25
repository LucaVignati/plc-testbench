from math import ceil, floor
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
        self.packet_size = settings.packet_size
        mpl.rcParams['agg.path.chunksize'] = 10000

    def plot_audio_track(self, node: Node, to_file=False) -> None:
        '''
        Plot the original input file
        '''
        audio_file = node.get_file()
        samplerate = audio_file.get_samplerate()
        n_channels = audio_file.get_channels()
        audio_file_data = audio_file.get_data()
        dots = len(audio_file_data)
        if dots > 500000:
            dots = 500000
        subsampling_factor = floor((len(audio_file_data)/dots))
        subsampled_audio_data = audio_file_data[::subsampling_factor]
        subsampled_samplerate = samplerate/subsampling_factor
        x = np.arange(0, len(subsampled_audio_data)/(subsampled_samplerate), 1/(subsampled_samplerate))

        fig, ax = plt.subplots(n_channels, 1, sharex=True, figsize=self.figsize, dpi=self.dpi)
        if issubclass(node.__class__, OriginalTrackNode):
            fig.suptitle("Original Track")
        elif issubclass(node.__class__, ECCTrackNode):
            fig.suptitle("ECC Track")

        for n in range(n_channels):
            if subsampled_audio_data.ndim > 1:
                audio_channel_data = subsampled_audio_data[:, n]
            else:
                audio_channel_data = subsampled_audio_data
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

        plt.close(fig)
    
    def plot_lost_samples_mask(self, node: LostSamplesMaskNode, to_file=False) -> None:
        '''
        Plot the lost samples mask data
        '''
        lost_packets_idx = node.get_file().get_data()[::self.packet_size]/self.packet_size
        original_track = node.get_original_track()
        samplerate = original_track.get_samplerate()
        original_track_length = (len(original_track.get_data()) - 1)/samplerate
        lost_packet_times = lost_packets_idx / (samplerate/self.packet_size)
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.suptitle("Lost Samples")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.vlines(lost_packet_times, 0, 1, linewidth=self.linewidth)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lost Samples")
        ax.set_xlim(0, original_track_length)
        if to_file:
            fig.savefig(node.get_path(), bbox_inches='tight')

        plt.close(fig)

    def plot_output_analysis(self, node: OutputAnalysisNode, to_file=False) -> None:
        '''
        Plot the output analysis data
        '''
        original_track = node.get_original_track()
        samplerate = original_track.get_samplerate()
        n_channels = original_track.get_channels()
        original_track_length = (len(original_track.get_data()) - 1)/samplerate
        data = node.get_file().get_data()
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        if issubclass(node.worker.__class__, MSECalculator):
            name = "Mean Square Error"
            fig.suptitle(name)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(name)
            ax.set_xlim(0, original_track_length)
            mse = data.get_mse()
            dots = len(mse)
            if dots > 500000:
                dots = 500000
            subsampling_factor = floor(len(mse)/dots)
            subsampled_mse = mse[::subsampling_factor]
            x = np.arange(0, original_track_length, original_track_length/dots)
            for n in range(n_channels):
                if mse.ndim > 1:
                    channel_data = subsampled_mse[:, n]
                else:
                    channel_data = subsampled_mse
                label = "Channel " + str(n + 1)
                ax.plot(x, channel_data, label=label)
            plt.legend(loc="upper left")
            if to_file:
                fig.savefig(node.get_path(), bbox_inches='tight')
        if issubclass(node.worker.__class__, SpectralEnergyCalculator):
            name = "Spectral Energy"
        if issubclass(node.worker.__class__, PEAQCalculator):
            name = "PEAQ"
            odg_text = "Objective Difference Grade: "
            di_text = "Distortion Index: "
            file_content = odg_text + str(data.get_odg()) + "\n" + di_text + str(data.get_di())
            print(file_content)
            if to_file:
                file = open(node.get_path(), "w")
                file.write(file_content)

        plt.close(fig)

    def show() -> None:
        plt.show()