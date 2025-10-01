import subprocess
import numpy as np
import numpy.random as npr
from .settings import Settings, PEAQMode
from .worker import Worker
from .file_wrapper import SimpleCalculatorData, PEAQData, AudioFile
from .utils import dummy_progress_bar, extract_intorni, force_single_loss_per_stimulus, is_loud_enough
from .perceptual_metric import *
from .listening_tests import ListeningTest
from .utils import fade_in, fade_out, leading_silence, trailing_silence
from .path_manager import _format_pls_settings
from .file_wrapper import AudioFile as AF

global call_count_PLC, call_count_PLS, call_count_audio
call_count_PLC = 0
call_count_PLS = 0
call_count_audio = 0

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
                print(f"Chunk {idx} too short for PEAQ, skipping.")
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
                    print(f"Could not parse PEAQ ODG value: {peaq_odg}")
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


class MultiHumanCalculator(OutputAnalyser):
    """
    Compares multiple PLC algorithms using a listening test for 1 Audio file (To Do: support multiple audio files).
    """
    _sessions: dict = {}

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.stimulus_length = self.settings.get("stimulus_length")
        self.no_audio_reuse_per_stimuli = self.settings.get("no_audio_reuse_per_stimuli")
        self.new_audio_per_page = self.settings.get("new_audio_per_page")
        self.pages_per_PLS = self.settings.get("pages_per_PLS")
        self.choose_seed = self.settings.get("choose_seed")
        self.original_audio_tracks = self.settings.get("original_audio_tracks")
        self.persistent = False
        self.plc_algorithms = self.settings.get("plc_algorithms")
        self.packet_loss_simulators = self.settings.get("packet_loss_simulators")

    def _transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def _session_key(self, original_track_node, lost_samples_idxs_data):
        from pathlib import Path
        return f"{Path(original_track_node.get_path()).stem}-{hash(self.settings)}-{len(lost_samples_idxs_data.get_data())}"

    def _select_or_get_indices(self, original_track_node, reconstructed_track_node, lost_samples_idxs_data, key):
        fs = self.fs
        if self.no_audio_reuse_per_stimuli:
            lost_samples_idxs = force_single_loss_per_stimulus(
                lost_samples_idxs_data.get_data(), fs, self.stimulus_length/2, self.packet_size)
        else:
            lost_samples_idxs = lost_samples_idxs_data.get_data()

        intorni_original = extract_intorni(original_track_node, lost_samples_idxs,
                                           self.stimulus_length, fs, self.packet_size, unique=True)
        intorni_reconstructed = extract_intorni(reconstructed_track_node, lost_samples_idxs,
                                                self.stimulus_length, fs, self.packet_size, unique=True)

        intorni_original_loud = []
        intorni_reconstructed_loud = []
        for idx in range(len(intorni_original[1])):
            if is_loud_enough(intorni_original[1][idx], original_track_node.get_data(), -10):
                intorni_original_loud.append([intorno[idx] for intorno in intorni_original])
                intorni_reconstructed_loud.append([intorno[idx] for intorno in intorni_reconstructed])

        if not intorni_original_loud:
            raise ValueError("No sufficiently loud stimuli were found.")

        intorni_original_loud = self._transpose(intorni_original_loud)
        intorni_reconstructed_loud = self._transpose(intorni_reconstructed_loud)

        session = self._sessions[key]
        if session["stimuli_indices"] is None:
            total_available = len(intorni_original_loud[1])
            np.random.seed(self.choose_seed)
            selected = list(range(total_available))
            if self.pages_per_PLS < len(selected):
                selected = selected[:self.pages_per_PLS]
            session["stimuli_indices"] = selected
            session["original_segments"] = self._transpose([[intorno[idx] for intorno in intorni_original_loud]
                                                            for idx in selected])
            session["original_full"] = intorni_original_loud
            session["stimuli_count"] = len(selected)

        algorithm_segments = self._transpose([[intorno[idx] for intorno in intorni_reconstructed_loud]
                                         for idx in session["stimuli_indices"]])
        return algorithm_segments

    @staticmethod
    def get_reconstructed_tracks_folder_name(simulator_class, simulator_settings):
        base = f"{simulator_class.__name__}-{_format_pls_settings(simulator_settings)}"
        return f"{base}"

    def _write_segments(self, listening_test, key, original_track_node, plc_algorithm, algorithm_segments):
        session = self._sessions[key]
        fs = original_track_node.get_samplerate()
        fade_time = 1

        simulator_class, simulator_settings = self.packet_loss_simulators[call_count_PLS]
        loss_pattern_folder = self.get_reconstructed_tracks_folder_name(simulator_class, simulator_settings)

        references_dir = listening_test.references_test_folder.joinpath(loss_pattern_folder)
        references_dir.mkdir(parents=True, exist_ok=True)
        stimuli_dir = listening_test.stimuli_test_folder.joinpath(loss_pattern_folder)
        stimuli_dir.mkdir(parents=True, exist_ok=True)

        # write reference files
        ref_paths = []
        for list_index, (packet_index, orig_seg) in enumerate(zip(session["original_segments"][0], session["original_segments"][1])):
            seg = orig_seg.copy()
            fade_in(seg, fs, fade_time)
            fade_out(seg, fs, fade_time)
            seg = leading_silence(seg, fs, 200)
            seg = trailing_silence(seg, fs, 300)
            if seg.ndim == 1:
                seg = np.stack([seg, seg], axis=-1)
            out_path = references_dir.joinpath(f"reference_{list_index}-{packet_index}.wav")
            AF.from_audio_file(original_track_node, new_data=seg, new_path=str(out_path))
            ref_paths.append(out_path)
        session["orig_ref_paths"].append(ref_paths)

        # write stimuli files
        stored_paths = []
        for list_index, (packet_index, recon_seg) in enumerate(zip(session["original_segments"][0], algorithm_segments[1])):
            seg = recon_seg.copy()
            fade_in(seg, fs, fade_time)
            fade_out(seg, fs, fade_time)
            seg = leading_silence(seg, fs, 200)
            seg = trailing_silence(seg, fs, 300)
            if seg.ndim == 1:
                seg = np.stack([seg, seg], axis=-1)
            out_path = stimuli_dir.joinpath(f"{plc_algorithm}_{list_index}-{packet_index}.wav")
            AF.from_audio_file(original_track_node, new_data=seg, new_path=str(out_path))
            stored_paths.append(out_path)
        if plc_algorithm not in session["recon_paths"]:
            session["recon_paths"][plc_algorithm] = []
        session["recon_paths"][plc_algorithm].append(stored_paths)

    def run(self, original_track_node: AudioFile, reconstructed_track_node: AudioFile, lost_samples_idxs_data):
        global call_count_PLC
        global call_count_PLS
        global call_count_audio

        key = "global"
        if key not in self._sessions:
            self._sessions[key] = {
                "expected_algos": None,
                "expected_algos_final": False,
                "stimuli_indices": None,
                "stimuli_count": None,
                "original_segments": None,
                "original_full": None,
                "written_original": False,
                "recon_paths": {},
                "stimuli_per_page": None,
                "listening_test": None,
                "processed_algos": 0,
                "orig_ref_paths": [],
            }
        session = self._sessions[key]
        session["lost_samples_idxs_data"] = lost_samples_idxs_data
        listening_test = ListeningTest(self.settings)
        plc_algorithms = self.plc_algorithms[call_count_PLC][0].__name__

        algorithm_segments = self._select_or_get_indices(original_track_node, reconstructed_track_node, lost_samples_idxs_data, key)
        self._write_segments(listening_test, key, original_track_node, plc_algorithms, algorithm_segments)

        number_of_audio = len(self.original_audio_tracks) - 1
        number_of_plc = len(self.plc_algorithms) - 1
        number_of_pls = len(self.packet_loss_simulators) - 1

        print(f"call_count_PLC: {call_count_PLC}, call_count_PLS: {call_count_PLS}, call_count_audio: {call_count_audio}")
        print(f"number_of_plc: {number_of_plc}, number_of_pls: {number_of_pls}, number_of_audio: {number_of_audio}")

        if call_count_PLC == number_of_plc:
            if call_count_PLS == number_of_pls:
                if call_count_audio == number_of_audio:
                    original_audio_track_names = [track[1].settings["filename"] for track in self.original_audio_tracks]
                    plc_algorithm_names = [plc_algorithm[0].__name__ for plc_algorithm in self.plc_algorithms]
                    packet_loss_simulators_names = [simulator[0].__name__ for simulator in self.packet_loss_simulators]
                    cfg_path = listening_test.generate_multi_config(session, original_audio_track_names, plc_algorithm_names,
                                                                    packet_loss_simulators_names, self.new_audio_per_page,
                                                                    self.new_audio_per_page, self.stimulus_length)
                    session["config_written"] = True
                    print(f"[MultiHumanCalculator] Created shared config: {cfg_path}")
                call_count_audio += 1 if call_count_audio <= number_of_audio else 0
            call_count_PLS = 0 if call_count_PLS == number_of_pls else call_count_PLS + 1
        call_count_PLC = 0 if call_count_PLC == number_of_plc else call_count_PLC + 1

        # Dummy-Metrik
        return np.full(len(original_track_node.get_data()) // self.packet_size, np.nan, dtype=float)