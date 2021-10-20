import numpy as np
from ecctestbench.worker import Worker

def normalise(x, amp_scale=1.0):
    return(amp_scale * x / np.amax(np.abs(x)))


class OutputAnalyser(Worker):

    def __str__(self) -> str:
        return __class__.__name__


class MSECalculator(OutputAnalyser):

    def run(self, original_track: np.ndarray, ecc_track: np.ndarray) -> None:
        '''
        Calculation of Mean Square Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                ecc_signal: N-length test signal array.

            Output:
                mse: Mean Square Error calculated between the two signals.
        '''
        amp_scale = self.settings.amp_scale
        N = self.settings.N
        hop = self.settings.hop

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(ecc_track, amp_scale)

        num_samples = len(x_r)

        w = np.hanning(N+1)[:-1]

        x_rw = np.array([w*x_r[i:i+N] for i in
                        range(0, num_samples-N, hop)])
        x_ew = np.array([w*x_e[i:i+N] for i in
                        range(0, num_samples-N, hop)])
        mse = [np.mean((x_rw[n] - x_ew[n])**2) for n in range(len(x_rw))]

        return mse

    def __str__(self) -> str:
        return __class__.__name__


class SpectralEnergyCalculator(OutputAnalyser):

    def run(self, original_track: np.ndarray, ecc_track: np.ndarray) -> None:
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
        amp_scale = self.settings.amp_scale
        N = self.settings.N
        hop = self.settings.hop

        w = np.hanning(N+1)[:-1]

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(ecc_track, amp_scale)

        num_samples = len(x_r)

        x_rk = np.array([np.fft.fft(w*x_r[i:i+N]) for i in
                        range(0, num_samples-N, hop)])
        x_ek = np.array([np.fft.fft(w*x_e[i:i+N]) for i in
                        range(0, num_samples-N, hop)])
        x_2rk = np.abs(x_rk)**2
        x_2ek = np.abs(x_ek)**2

        se = np.array(x_2rk - 2*np.sqrt(x_2rk * x_2ek) + x_2ek)

        return se

    def __str__(self) -> str:
        return __class__.__name__

class PEAQCalculator(OutputAnalyser):

    def run(self, original_track: np.ndarray, ecc_track: np.ndarray) -> None:
        pass

    def __str__(self) -> str:
        return __class__.__name__