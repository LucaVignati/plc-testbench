import numpy as np


def normalise(x, amp_scale=1.0):
    return(amp_scale * x / np.amax(np.abs(x)))


class OutPutAnalyser(object):

    def __init__(self, buffer_size, amp_scale=1.0, fs=48000, N=1024):
        '''
        Base Class Initialisation for the Output Analyser classes

            Input:
                Fs: Sample Rate
                Buffer Size:
                N: Window Size
                amp_scale: Maximum Amplitude Scalar for normalisation

        '''
        self._fs = fs
        self._buffer_size = buffer_size
        self._N = N
        self._hop = N//2
        self._amp_scale = amp_scale


class MSECalculator(OutPutAnalyser):

    def run(self, ref_signal, ecc_signal):
        '''
        Calculation of Mean Square Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                ecc_signal: N-length test signal array.

            Output:
                mse: Mean Square Error calculated between the two signals.
        '''
        x_r = normalise(ref_signal, self._amp_scale)
        x_e = normalise(ecc_signal, self._amp_scale)

        num_samples = len(x_r)

        w = np.hanning(self._N+1)[:-1]

        x_rw = np.array([w*x_r[i:i+self._N] for i in
                        range(0, num_samples-self._N, self._hop)])
        x_ew = np.array([w*x_e[i:i+self._N] for i in
                        range(0, num_samples-self._N, self._hop)])
        mse = [np.mean((x_rw[n] - x_ew[n])**2) for n in range(len(x_rw))]

        return mse


class SpectralEnergyCalculator(OutPutAnalyser):

    def run(self, ref_signal, ecc_signal):
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

        w = np.hanning(self._N+1)[:-1]

        x_r = normalise(ref_signal, self._amp_scale)
        x_e = normalise(ecc_signal, self._amp_scale)

        num_samples = len(x_r)

        x_rk = np.array([np.fft.fft(w*x_r[i:i+self._N]) for i in
                        range(0, num_samples-self._N, self._hop)])
        x_ek = np.array([np.fft.fft(w*x_e[i:i+self._N]) for i in
                        range(0, num_samples-self._N, self._hop)])
        x_2rk = np.abs(x_rk)**2
        x_2ek = np.abs(x_ek)**2

        se = np.array(x_2rk - 2*np.sqrt(x_2rk * x_2ek) + x_2ek)

        return se
