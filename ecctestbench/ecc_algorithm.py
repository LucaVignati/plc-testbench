from typing import Tuple
import numpy as np
from ecctestbench.data_manager import DataManager
from .node import Node

class ECCAlgorithm(object):

    def __init__(self, settings):

        self._buffer_size = settings.buffer_size

    def __str__(self) -> str:
        return __class__.__name__


class ZerosEcc(ECCAlgorithm):

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        Run the ECC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        ECC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''
        ecc_track = original_track
        for idx in lost_samples_idx:
            ecc_track[idx] = 0

        return ecc_track

    def __str__(self) -> str:
        return __class__.__name__


class LastPacketEcc(ECCAlgorithm):

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        Run the ECC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        ECC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''

        ecc_track = original_track
        for idx in lost_samples_idx:
            ecc_track[idx] = 0
        # for n, x in enumerate(output_wave):
        #     n

        return ecc_track

    def __str__(self) -> str:
        return __class__.__name__


# class LMSRegressionECC(ECCAlgorithm):

#     def run(self, input_wave, lost_packet_mask):
#         '''

#         '''
#         output_wave = input_wave * lost_packet_mask

#         return output_wave
