from sys import path
from typing import Set
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from ecctestbench.path_manager import PathManager
from .settings import Settings
from .data_manager import DataManager

class PlotManager(object):

    def __init__(self, settings: Settings, rows:int = None, cols:int = None) -> None:
        '''
        Base class for plotting results

        '''
        self.N = settings.N
        self.hop = settings.hop
        self.rows = rows
        self.cols = cols

    def plot_audio_track(self, audio_track: sf.SoundFile, show=True, to_file=False, to_subplots=False) -> None:
        '''
        Plot the original input file
        '''
        plt.plot(audio_track.read())
        if show:
            plt.show()
        if to_file:
            plt.savefig(PathManager.change_file_extension(audio_track.name, 'png'))


    def plot_mse(self) -> None:
        '''
        Plot the mse and the number of lost samples per window
        '''
        lost_packet_mask = self.data_manager.get_lost_packet_mask()
        output_measurement = self.data_manager.get_output_measurement()

        num_samples = len(lost_packet_mask)

        lost_packet_mask = np.logical_not(lost_packet_mask).astype(int)

        lost_packets_per_window = np.array([sum(lost_packet_mask[i:i+self.N]) for i in
                                    range(0, num_samples-self.N, self.hop)])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(output_measurement[0])
        #ax1.xticks(np.arange(0, len(mse), step=128))
        ax1.set_xlabel('time')
        ax1.set_ylabel('mse')
        ax1.grid(True)

        ax2.bar(range(0, len(lost_packets_per_window)), lost_packets_per_window)
        ax2.set_xlabel('time')
        ax2.set_ylabel('lost samples per window')
        ax2.grid(True)

        plt.show()