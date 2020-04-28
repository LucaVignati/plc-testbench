import numpy as np


class output_plugins(object):

    def __init__(self, buffer_size, amax=1, fs=44100):

        self.fs = fs
        self.buffer_size = buffer_size
        self.adv = self.buffer_size/2
        self.amax = amax

    def compute_mse(self, ref_signal, ecc_signal):

        '''
            Calculation of Mean Square Error between the reference and signal \
                under test

            ref_signal: original N-length signal array

            ecc_signal: test signal
        '''

        self.sig_ref = ref_signal
        self.sig_ecc = ecc_signal

        error = self.sig_ref - self.sig_ecc

        sqrd_err = np.square(error)

        mse = np.mean(sqrd_err)

        return mse
