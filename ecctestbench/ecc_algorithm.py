import numpy as np
from ecctestbench.worker import Worker

class ECCAlgorithm(Worker):

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
        ecc_track = np.array(original_track)
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

        ecc_track = np.array(original_track)
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
