import numpy as np
import numpy.random as npr
from ecctestbench.worker import Worker

class PacketLossSimulator(Worker):
    '''
    Base class for the Packet Loss Generator that uses a
    range of statistical methods to generate an an N-length masking array
    of 1's (valid sample) and 0's (dropped sample), to simulate random
    packet loss in an audio signal. The 1's and 0's represent binary
    scalars that are mixed with the respective samples indexes of the
    input audio signal.
    '''

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.settings.seed)


class BasePacketLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int):

        return []

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.settings.seed)


class BinomialSampleLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int) -> np.ndarray:
        '''
        Generate a mask consisting of randomly distributed 1's or 0's across
        the whole N-length.

            Input:
                num_samples: The N-length of the input signal under test.

            Output:
                lost_packet_mask: N-length array where individual values
                are either 1 (valid sample) or 0 (dropped sample).
        '''
        per = self.settings.per
        npr.seed(self.settings.seed)
        lost_samples_mask = npr.choice(2, num_samples,
                                      p=[per, 1.0 - per])
        lost_samples_idx = [i for i in range(len(lost_samples_mask)) if lost_samples_mask[i]==0]

        return np.array(lost_samples_idx)

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.settings.seed)


class BinomialPacketLossSimulator(PacketLossSimulator):

    def run(self, num_samples: int) -> np.ndarray:
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
        per = self.settings.per
        npr.seed(self.settings.seed)
        Nb = int(num_samples//self.settings.buffer_size)

        lost_samples_mask = npr.choice(2, Nb+1, p=[per, 1.0 - per])
        lost_buffers = [np.ones(self.settings.buffer_size, dtype=int) *
                        b for b in lost_samples_mask]
        lost_samples_mask = np.concatenate(lost_buffers)
        lost_samples_mask = lost_samples_mask[0:num_samples]

        lost_samples_idx = [i for i in range(len(lost_samples_mask)) if lost_samples_mask[i]==0]

        return np.array(lost_samples_idx)

    def __str__(self) -> str:
        return __class__.__name__ + '_' + str(self.settings.seed)