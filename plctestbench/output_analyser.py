import subprocess
import numpy as np
import numpy.random as npr
from .settings import Settings, PEAQMode
from .worker import Worker
from .file_wrapper import SimpleCalculatorData, PEAQData, AudioFile
from .utils import dummy_progress_bar, extract_intorni, force_single_loss_per_stimulus, is_loud_enough
from .perceptual_metric import *
from .listening_tests import ListeningTest

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

        return SimpleCalculatorData(np.array([x_rw, x_ew]))


class MSECalculator(SimpleCalculator):
    '''
    MSECalculator is ...
    '''
    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs = None):
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
        return SimpleCalculatorData(np.array(error))


class MAECalculator(SimpleCalculator):
    '''
    MAECalculator is ...
    '''
    
    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs = None) -> SimpleCalculatorData:
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
        return SimpleCalculatorData(np.array(error))


class SpectralEnergyCalculator(OutputAnalyser):
    '''
    SpectralEnergyCalculator is ...
    '''
    
    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs = None) -> SimpleCalculatorData:
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

        fft_results = [(np.fft.fft(w*x_r[i:i+N]), np.fft.fft(w*x_e[i:i+N])) for i in
               self.progress_monitor(range(0, num_samples-N, hop), desc=str(self))]

        x_rk, x_ek = map(list, zip(*fft_results))
        x_2rk = np.abs(np.array(x_rk))**2
        x_2ek = np.abs(np.array(x_ek))**2

        se = np.array(x_2rk - 2*np.sqrt(x_2rk * x_2ek) + x_2ek)

        return SimpleCalculatorData(se)


class PEAQCalculator(OutputAnalyser):
    '''
    PEAQCalculator is ...
    '''
    
    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs = None) -> PEAQData:
        peaq_mode = self.settings.get("peaq_mode")
        if peaq_mode == PEAQMode.basic.value:
            mode_flag = '--basic'
        elif peaq_mode == PEAQMode.advanced.value:
            mode_flag = '--advanced'
        else:
            mode_flag = ''
        path = original_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(original_track_node.get_data())
        original_track_norm_file = AudioFile.from_audio_file(original_track_node, new_data=new_data, new_path=new_path)
        original_track_norm_file.save()
        path = reconstructed_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(reconstructed_track_node.get_data())
        reconstructed_track_norm_file = AudioFile.from_audio_file(reconstructed_track_node, new_data=new_data, new_path=new_path)
        reconstructed_track_norm_file.save()

        if mode_flag == '':
            completed_process = subprocess.run(["peaq", "--gst-plugin-path", "/usr/lib/gstreamer-1.0/", original_track_norm_file.get_path(),
                                                reconstructed_track_norm_file.get_path()], capture_output=True, text=True, check=False)
        else:
            completed_process = subprocess.run(["peaq", mode_flag, "--gst-plugin-path", "/usr/lib/gstreamer-1.0/", original_track_norm_file.get_path(),
                                                reconstructed_track_norm_file.get_path()], capture_output=True, text=True, check=False)

        original_track_norm_file.delete()
        reconstructed_track_norm_file.delete()

        peaq_output = completed_process.stdout

        dummy_progress_bar(self)

        peaq_odg_text = "Objective Difference Grade: "
        peaq_di_text = "Distortion Index: "
        if (peaq_odg_text in peaq_output and peaq_di_text in peaq_output):
            peaq_odg, peaq_di = peaq_output.split("\n", 1)
            _, peaq_odg = peaq_odg.split(peaq_odg_text)
            _, peaq_di = peaq_di.split(peaq_di_text)
            peaq_odg = float(peaq_odg)
            peaq_di = float(peaq_di)
            return PEAQData(peaq_odg, peaq_di)
        else:
            print("The peaq program exited with the following errors:")
            print(completed_process.stdout)
            # Return a default PEAQData object in case of error
            return PEAQData(float('nan'), float('nan'))

class WindowedPEAQCalculator(OutputAnalyser):
    '''
    WindowedPEAQCalculator is ...
    '''

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.intorno_length = self.settings.get("intorno_length")
        self.mode_flag = ''
        self.sign = 1
        peaq_mode = self.settings.get("peaq_mode")
        if peaq_mode == PEAQMode.basic.value:
            self.mode_flag = '--basic'
            self.sign = -1
        elif peaq_mode == PEAQMode.advanced.value:
            self.mode_flag = '--advanced'

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs_data) -> SimpleCalculatorData:
        path = original_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(original_track_node.get_data())
        original_track_norm_file = AudioFile.from_audio_file(original_track_node, new_data=new_data, new_path=new_path)
        original_track_norm_file.save()
        path = reconstructed_track_node.get_path()
        new_path = path[:-4] + "_norm" + path[-4:]
        new_data = normalise(reconstructed_track_node.get_data())
        reconstructed_track_norm_file = AudioFile.from_audio_file(reconstructed_track_node, new_data=new_data, new_path=new_path)
        reconstructed_track_norm_file.save()

        lost_samples_idxs = lost_samples_idxs_data.get_data()
        intorni_original = extract_intorni(original_track_norm_file, lost_samples_idxs, self.intorno_length, self.fs, self.packet_size)
        intorni_reconstructed = extract_intorni(reconstructed_track_norm_file, lost_samples_idxs, self.intorno_length, self.fs, self.packet_size)

        path = original_track_node.get_path()
        original_path = path[:-4] + "_chunk" + path[-4:]
        path = reconstructed_track_node.get_path()
        reconstructed_path = path[:-4] + "_chunk" + path[-4:]
        original_data = original_track_node.get_data()
        if original_data is None:
            raise ValueError("original_track_node.get_data() returned None.")
        metric = np.zeros(len(original_data) // self.packet_size)

        for idx, (intorno_original, intorno_reconstructed) in enumerate(zip(intorni_original[1], intorni_reconstructed[1])):
            # Prüfe Chunk-Länge
            if len(intorno_original) < int(self.fs * 0.4):
                print(f"Chunk {idx} zu kurz für PEAQ, wird übersprungen.")
                metric[idx] = np.nan
                continue

            original_intorno_file = AudioFile.from_audio_file(original_track_norm_file, new_data=intorno_original, new_path=original_path)
            reconstructed_intorno_file = AudioFile.from_audio_file(reconstructed_track_norm_file, new_data=intorno_reconstructed, new_path=reconstructed_path)
            original_intorno_file.save()
            reconstructed_intorno_file.save()

            completed_process = subprocess.run(
                ["peaq", self.mode_flag, "--gst-plugin-path", "/usr/lib/gstreamer-1.0/",
                original_intorno_file.get_path(), reconstructed_intorno_file.get_path()],
                capture_output=True, text=True, check=False)

            original_intorno_file.delete()
            reconstructed_intorno_file.delete()

            peaq_output = completed_process.stdout

            peaq_odg_text = "Objective Difference Grade: "
            peaq_di_text = "Distortion Index: "
            if (peaq_odg_text in peaq_output and peaq_di_text in peaq_output):
                peaq_odg, _ = peaq_output.split("\n", 1)
                _, peaq_odg = peaq_odg.split(peaq_odg_text)
                try:
                    metric[idx] = self.sign * float(peaq_odg)
                    print(f"metric[{idx}] set to {metric[idx]} (parsed ODG: {peaq_odg})")
                except ValueError:
                    print(f"PEAQ ODG konnte nicht geparst werden: {peaq_odg}")
                    metric[idx] = np.nan
            else:
                print("The peaq program exited with the following errors:")
                print(completed_process.stdout)
                metric[idx] = np.nan

        original_track_norm_file.delete()
        reconstructed_track_norm_file.delete()

        return SimpleCalculatorData(metric)
        
class PerceptualCalculator(OutputAnalyser):
    '''
    PerceptualCalculator is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.intorno_length = self.settings.get("intorno_length")
        self.linear_mag = self.settings.get("linear_mag") 
        self.transform_type = self.settings.get("transform_type") 
        self.min_frequency = self.settings.get("min_frequency")
        self.max_frequency_perceptual = self.settings.get("max_frequency_perceptual")
        self.bins_per_octave = self.settings.get("bins_per_octave")
        self.n_bins = self.settings.get("n_bins")
        self.minimum_window = self.settings.get("minimum_window")
        self.masking = self.settings.get("masking")
        self.masking_offset = self.settings.get("masking_offset")
        self.db_weighting = self.settings.get("db_weighting")
        self.metric = self.settings.get("metric")

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs_data):
        lost_samples_idxs = lost_samples_idxs_data.get_data()
        intorni_original = extract_intorni(original_track_node, lost_samples_idxs, self.intorno_length, self.fs, self.packet_size)
        intorni_reconstructed = extract_intorni(reconstructed_track_node, lost_samples_idxs, self.intorno_length, self.fs, self.packet_size)
        
        if intorni_original[1][0].ndim == 1:
            input_size = len(intorni_original[1][0])
        else:
            input_size = len(intorni_original[1][0][:, 0])

        pm = PerceptualMetric(self.transform_type,
                              self.min_frequency,
                              self.max_frequency_perceptual,
                              self.bins_per_octave,
                              self.n_bins,
                              self.minimum_window,
                              input_size,
                              self.fs,
                              self.intorno_length,
                              self.linear_mag,
                              self.masking,
                              self.masking_offset,
                              self.db_weighting,
                              self.metric)

        spectrograms = []
        for idx, original, reconstructed in self.progress_monitor(zip(intorni_original[0], intorni_original[1], intorni_reconstructed[1]),
                                                                  total=len(intorni_original[1]), desc=str(self)):
            if intorni_original[1][0].ndim == 1:
                spectrograms.append({'idx': idx, **pm.spectrogram(original[:], reconstructed[:])})
            else:
                for channel in range(original.shape[1]):
                    spectrograms.append({'idx': (idx, channel), **pm.spectrogram(original[:, channel], reconstructed[:, channel])})
                    
        if intorni_original[1][0].ndim == 1:
            metric = np.zeros(len(original_track_node.get_data()) // self.packet_size)
        else:
            metric = np.zeros((len(original_track_node.get_data()) // self.packet_size, 2))

        for spectrogram in spectrograms:
            perc_metric = pm(spectrogram)
            if isinstance(spectrogram['idx'], tuple):
                idx, channel = spectrogram['idx']
                metric[idx, channel] = perc_metric
            else:
                metric[spectrogram['idx']] = perc_metric

        return SimpleCalculatorData(metric)
    
class HumanCalculator(OutputAnalyser):
    '''
    ListeningTest is ...
    '''

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.stimulus_length = self.settings.get("stimulus_length")
        self.single_loss = self.settings.get("single_loss_per_stimulus")
        self.stimuli_per_page = self.settings.get("stimuli_per_page")
        self.pages = self.settings.get("pages")
        self.stimuli_number = self.stimuli_per_page * self.pages
        self.choose_seed = self.settings.get("choose_seed")
        self.persistent = False

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs_data):

        def transpose(matrix):
            return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

        self.listening_test = ListeningTest(self.settings)

        if self.single_loss:
            lost_samples_idxs = force_single_loss_per_stimulus(lost_samples_idxs_data.get_data(), self.fs, self.stimulus_length/2, self.packet_size)
        else:
            lost_samples_idxs = lost_samples_idxs_data.get_data()
        intorni_original = extract_intorni(original_track_node, lost_samples_idxs, self.stimulus_length, self.fs, self.packet_size, unique=True)
        intorni_reconstructed = extract_intorni(reconstructed_track_node, lost_samples_idxs, self.stimulus_length, self.fs, self.packet_size, unique=True)

        intorni_original_loud = []
        intorni_reconstructed_loud = []
        for idx in range(len(intorni_original[1])):
            if is_loud_enough(intorni_original[1][idx], original_track_node.get_data(), -10):
                intorni_original_loud.append([intorno[idx] for intorno in intorni_original])
                intorni_reconstructed_loud.append([intorno[idx] for intorno in intorni_reconstructed])

        intorni_original_loud = transpose(intorni_original_loud)
        intorni_reconstructed_loud = transpose(intorni_reconstructed_loud)

        if self.stimuli_number > len(intorni_original_loud[1]):
            error_message = (f"The number of stimuli requested ({self.stimuli_number}) is greater than the number of stimuli available "
                             f"({len(intorni_original_loud)}). Increase the total length of available audio.")
            discarded_packets_close = (len(lost_samples_idxs_data.get_data()) - len(lost_samples_idxs)) // self.packet_size
            if discarded_packets_close > 0:
                error_message += f" {discarded_packets_close} stimulus were discarded because too close to each other."
            discarded_packet_loud = len(intorni_original[1]) - len(intorni_original_loud[1])
            if discarded_packet_loud > 0:
                error_message += f" {discarded_packet_loud} stimulus were discarded because too quiet."
            raise ValueError(error_message)
        

        npr.seed(self.choose_seed)
        stimuli_idxs = npr.choice(range(len(intorni_original_loud[1])), self.stimuli_number, replace=False)
        stimuli_original = transpose([[intorno[idx] for intorno in intorni_original_loud] for idx in stimuli_idxs])
        stimuli_reconstructed = transpose([[intorno[idx] for intorno in intorni_reconstructed_loud] for idx in stimuli_idxs])

        self.listening_test.set_references(list(zip(stimuli_original[0], stimuli_original[1])), original_track_node)
        self.listening_test.set_stimuli(list(zip(stimuli_reconstructed[0], stimuli_reconstructed[1])), original_track_node)
        self.listening_test.set_indexes(stimuli_original[0])
        self.listening_test.generate_config()
        try:
            results = self.listening_test.get_results()
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            choice = input("Human-Calculator Results are missing. Skip? (y=yes / n=no) ").strip().lower()
            if choice == 'y':
                return np.full(len(original_track_node.get_data())//self.packet_size, np.nan, dtype=float)
            raise
        results =  self.listening_test.get_results()

        metric = np.zeros(len(original_track_node.get_data())//self.packet_size)

        string_to_int_map = {}
        next_available_index = len(metric)  # Startindex für Strings (nach den numerischen Indizes)

        for idx, mean, _ in self.progress_monitor(results, desc=str(self)):
            try:
                index = int(idx.split('-')[-1])
            except ValueError:
                if idx not in string_to_int_map:
                    string_to_int_map[idx] = next_available_index
                    next_available_index += 1
                index = string_to_int_map[idx]          
            if index >= len(metric):
                metric = np.pad(metric, (0, index - len(metric) + 1), 'constant', constant_values=0)
            metric[index] = mean

        return SimpleCalculatorData(metric)