import subprocess
from time import sleep
import numpy as np
from .settings import Settings
from .worker import Worker
from .file_wrapper import SimpleCalculatorData, PEAQData, AudioFile

def normalise(x, amp_scale=1.0):
    return(amp_scale * x / np.amax(np.abs(x)))


class OutputAnalyser(Worker):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

class SimpleCalculator(OutputAnalyser):

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile) -> SimpleCalculatorData:
        '''
        Calculation of Mean Square Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                reconstructed_signal: N-length test signal array.

            Output:
                x_rw: N-length array of windowed reference signal frames.
                x_ew: N-length array of windowed test signal frames.
        '''
        amp_scale = self.settings.get("amp_scale")
        N = self.settings.get("N")
        hop = self.settings.get("hop")
        original_track = original_track_node.get_data()
        reconstructed_track = reconstructed_track_node.get_data()

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(reconstructed_track, amp_scale)

        num_samples = len(x_r)

        w = np.hanning(N+1)[:-1]
        if x_r.ndim > 1:
            w = np.transpose(np.tile(w, (np.shape(x_r)[1], 1)))

        x_rw = np.array([np.multiply(w, x_r[i:i+N]) for i in
                        range(0, num_samples-N, hop)])
        x_ew = np.array([np.multiply(w, x_e[i:i+N]) for i in
                        range(0, num_samples-N, hop)])

        return x_rw, x_ew

class MSECalculator(SimpleCalculator):

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile):
        '''
        Calculation of Mean Square Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                reconstructed_signal: N-length test signal array.

            Output:
                error: Mean Square Error calculated calculated between the two signals.
        '''
        x_rw, x_ew = super().run(original_track_node, reconstructed_track_node)
        error = [np.mean((x_rw[n] - x_ew[n])**2, 0) for n in self.progress_monitor(range(len(x_rw)), desc=str(self))]
        return SimpleCalculatorData(error)

class MAECalculator(SimpleCalculator):

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile):
        '''
        Calculation of Mean Absolute Error between the reference and signal
        under test.

            Input:
                ref_signal: original N-length signal array.
                reconstructed_signal: N-length test signal array.

            Output:
                error: Mean Absolute Error calculated calculated between the two signals.
        '''
        x_rw, x_ew = super().run(original_track_node, reconstructed_track_node)
        error = [np.mean(np.abs((x_rw[n] - x_ew[n])), 0) for n in self.progress_monitor(range(len(x_rw)), desc=str(self))]
        return SimpleCalculatorData(error)

class SpectralEnergyCalculator(OutputAnalyser):

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile):
        '''
        Calculate a difference magnitude signal from the DFT energies of the
        reference and signal under test.

            Input:
                ref_signal: original N-length signal array.
                reconstructed_signal: N-length test signal array.

            Output:
                se: Difference Magnitude signal array calulated from the
                Short-Time spectral differences between the reference and test.
        '''
        amp_scale = self.settings.get("amp_scale")
        N = self.settings.get("N")
        hop = self.settings.get("hop")
        original_track = original_track_node.get_data()
        reconstructed_track = reconstructed_track_node.get_data()

        w = np.hanning(N+1)[:-1]

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(reconstructed_track, amp_scale)

        num_samples = len(x_r)

        x_rk, x_ek = [(np.fft.fft(w*x_r[i:i+N]), np.fft.fft(w*x_e[i:i+N])) for i in
                        self.progress_monitor(range(0, num_samples-N, hop), desc=str(self))]
        x_2rk = np.abs(np.array(x_rk))**2
        x_2ek = np.abs(np.array(x_ek))**2

        se = np.array(x_2rk - 2*np.sqrt(x_2rk * x_2ek) + x_2ek)

        return SimpleCalculatorData(se)

class PEAQCalculator(OutputAnalyser):

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile) -> PEAQData:
        peaq_mode = self.settings.get("peaq_mode")
        if peaq_mode == 'basic':
            mode_flag = '--basic'
        elif peaq_mode == 'advanced':
            mode_flag = '--advanced'
        else:
            mode_flag = ''
        # print("GSTREAMER PEAQ running...", end=" ")
        path = original_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(original_track_node.get_data())
        original_track_norm_file = AudioFile.from_audio_file(original_track_node, new_data=new_data, new_path=new_path)
        path = reconstructed_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(reconstructed_track_node.get_data())
        reconstructed_track_norm_file = AudioFile.from_audio_file(reconstructed_track_node, new_data=new_data, new_path=new_path)
        completed_process = subprocess.run(["peaq", mode_flag, "--gst-plugin-path", "/usr/lib/gstreamer-1.0/",
                                           original_track_norm_file.get_path(), reconstructed_track_norm_file.get_path()], capture_output=True, text=True, check=False)

        original_track_norm_file.delete()
        reconstructed_track_norm_file.delete()
        # print("Completed.")

        peaq_output = completed_process.stdout

        for _ in self.progress_monitor(range(1), desc=str(self)):
            sleep(0.1)

        peaq_odg_text = "Objective Difference Grade: "
        peaq_di_text = "Distortion Index: "
        if (peaq_odg_text in peaq_output and peaq_di_text in peaq_output):
            peaq_odg, peaq_di = peaq_output.split("\n", 1)
            _, peaq_odg = peaq_odg.split(peaq_odg_text)
            _, peaq_di = peaq_di.split(peaq_di_text)
            peaq_odg = float(peaq_odg)
            peaq_di = float(peaq_di)
            return PEAQData(peaq_odg, peaq_di)

        print("The peaq program exited with the following errors:")
        print(completed_process.stdout)
        