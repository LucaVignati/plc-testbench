from typing import Tuple
import numpy as np
from ecctestbench.data_manager import DataManager
from .node import Node

def retrieve_data(node: Node) -> Tuple[np.ndarray, np.ndarray]:
    original_track_data = DataManager.get_original_track(node).read()
    ecc_track_data = DataManager.get_original_track(node).read()
    return original_track_data, ecc_track_data

def normalise(x, amp_scale=1.0):
    return(amp_scale * x / np.amax(np.abs(x)))


class OutputAnalyser(object):

    def __init__(self, settings):
        '''
        Base Class Initialisation for the Output Analyser classes

            Input:
                Fs: Sample Rate
                Buffer Size:
                N: Window Size
                amp_scale: Maximum Amplitude Scalar for normalisation

        '''
        self._fs = settings.fs
        self._buffer_size = settings.buffer_size
        self._N = settings.N
        self._hop = settings.N//2
        self._amp_scale = settings.amp_scale

    def __str__(self) -> str:
        return __class__.__name__


class MSECalculator(OutputAnalyser):

    def run(self, node: Node) -> None:
        '''
        Calculation of Mean Square Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                ecc_signal: N-length test signal array.

            Output:
                mse: Mean Square Error calculated between the two signals.
        '''
        original_track, ecc_track = retrieve_data(node)

        x_r = normalise(original_track, self._amp_scale)
        x_e = normalise(ecc_track, self._amp_scale)

        num_samples = len(x_r)

        w = np.hanning(self._N+1)[:-1]

        x_rw = np.array([w*x_r[i:i+self._N] for i in
                        range(0, num_samples-self._N, self._hop)])
        x_ew = np.array([w*x_e[i:i+self._N] for i in
                        range(0, num_samples-self._N, self._hop)])
        mse = [np.mean((x_rw[n] - x_ew[n])**2) for n in range(len(x_rw))]

        DataManager.store_data(node, mse)

    def __str__(self) -> str:
        return __class__.__name__


class SpectralEnergyCalculator(OutputAnalyser):

    def run(self, node: Node) -> None:
        '''
        Calculate a difference magnitude signal from the DFT energies of the
        reference and signal under test.

            Input:
                ref_signal: original N-length signal array.
                ecc_signal: N-length test signal array.

            Output:
                se: Difference Magnitude signal array calulated from the
                Short-Time spectral differences between the reference and test.
        '''
        original_track, ecc_track = retrieve_data(node)

        w = np.hanning(self._N+1)[:-1]

        x_r = normalise(original_track, self._amp_scale)
        x_e = normalise(ecc_track, self._amp_scale)

        num_samples = len(x_r)

        x_rk = np.array([np.fft.fft(w*x_r[i:i+self._N]) for i in
                        range(0, num_samples-self._N, self._hop)])
        x_ek = np.array([np.fft.fft(w*x_e[i:i+self._N]) for i in
                        range(0, num_samples-self._N, self._hop)])
        x_2rk = np.abs(x_rk)**2
        x_2ek = np.abs(x_ek)**2

        se = np.array(x_2rk - 2*np.sqrt(x_2rk * x_2ek) + x_2ek)

        DataManager.store_data(node, se)

    def __str__(self) -> str:
        return __class__.__name__

class PEAQCalculator(OutputAnalyser):

    def run(self, node: Node) -> None:
        pass

    def __str__(self) -> str:
        return __class__.__name__