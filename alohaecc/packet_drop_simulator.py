import numpy as np
from numpy.random import choice


class packet_drop_simulator(object):

    def __init__(self, buffer_size):
        self._buffer_size = buffer_size

    def generate_lost_packet_mask(self, num_samples):

        """
        Generate an N-length masking array of 1's (valid sample) and 0's \
            (dropped sample), to simulate random packet loss in an audio \
                signal. The 1's and 0's represent binary scalars that are \
                    mixed with the respective samples of the input audio \
                        signal.

            Input:
                num_samples: The N-length of the input signal under test

            Output:
                lost_packet_mask: N-length array where values are either 1 \
(valid sample) or 0 (dropped sample)
        """

        lost_packet_mask = choice(2, num_samples)

        return lost_packet_mask
