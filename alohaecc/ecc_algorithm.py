import numpy as np


class ECCAlgorithm(object):

    def __init__(self, buffer_size):

        self._buffer_size = buffer_size


class ZerosEcc(ECCAlgorithm):

    def run(self, input_wave, lost_packet_mask):
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
        output_wave = lost_packet_mask * input_wave

        return output_wave


class LastPacketEcc(ECCAlgorithm):

    def run(self, input_wave, lost_packet_mask):
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
        output_wave = np.array(input_wave) * lost_packet_mask
        for n, x in enumerate(output_wave):
            n

        return output_wave


class LMSRegressionECC(ECCAlgorithm):

    def run(self, input_wave, lost_packet_mask):
        '''

        '''
        output_wave = input_wave + lost_packet_mask

        return output_wave
