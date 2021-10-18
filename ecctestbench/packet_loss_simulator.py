import numpy as np
import numpy.random as npr
from ecctestbench.data_manager import DataManager
from .node import Node

class PacketLossSimulator(object):
    '''
    Base class for the Packet Loss Generator that uses a
    range of statistical methods to generate an an N-length masking array
    of 1's (valid sample) and 0's (dropped sample), to simulate random
    packet loss in an audio signal. The 1's and 0's represent binary
    scalars that are mixed with the respective samples indexes of the
    input audio signal.
    '''

    def __init__(self, settings):
        '''
        Base class initialisation for the Packet loss simulator

            Input:
                buffer_size: Buffer size.
                per: probabilities associated with each possible randomly
                generated outcome. This number represents the probability
                of a dropped sample (0) within the given array, and the
                inverse of this represents the probability of a valid sample
                (1).

        '''
        self._buffer_size = settings.buffer_size
        self._per = settings.per
        self.seed = settings.seed

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.seed)


class BasePacketLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int):

        lost_samples_mask = np.ones(num_samples)

        return lost_samples_mask

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.seed)


class BinomialSampleLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int):
        '''
        Generate a mask consisting of randomly distributed 1's or 0's across
        the whole N-length.

            Input:
                num_samples: The N-length of the input signal under test.

            Output:
                lost_packet_mask: N-length array where individual values
                are either 1 (valid sample) or 0 (dropped sample).
        '''
        lost_samples_mask = npr.choice(2, num_samples,
                                      p=[self._per, 1.0 - self._per])

        return lost_samples_mask

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.seed)


class BinomialPacketLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int):
        '''
        Generate a mask that drops whole buffers within an audio sample.
        This allocates fills each number of buffer lengths spanning the
        given N-length.

            Input:
                num_samples: The N-length of the input signal under test.

            Output:
                lost_packet_mask: N-length array where values within each
                buffer are either 1 (valid sample) or 0 (dropped sample).
        '''
        Nb = int(num_samples//self._buffer_size)

        lost_samples_mask = npr.choice(2, Nb+1, p=[self._per, 1.0 - self._per])
        lost_buffers = [np.ones(self._buffer_size, dtype=int) *
                        b for b in lost_samples_mask]
        lost_samples_mask = np.concatenate(lost_buffers)
        lost_samples_mask = lost_samples_mask[0:num_samples]

        return lost_samples_mask

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.seed)