from math import floor
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class LowCostConcealment:
    '''
    This class implements the Low Cost Concealment (LCC) described
    in "Low-delay error concealment with low computational overhead
    for audio over ip applications" by Marco Fink and Udo Zölzer
    '''

    def __init__(self, max_frequency: float, f_min: float, beta: float, n_m: int, fade_in_length: int, fade_out_length: int, extraction_length: int) -> None:
        self._max_frequency = max_frequency
        self._f_min = f_min
        self._beta = beta
        self._n_m = n_m
        self._fade_in_length = fade_in_length
        self._fade_out_length = fade_out_length
        self._extraction_length = extraction_length
        self._crossfade = False

    def prepare_to_play(self, samplerate: int, packet_size: int, n_channels: int):
        self._samplerate = samplerate
        self._packet_size = packet_size
        self._n_channels = n_channels
        self._win_size = round((self._beta * self._samplerate) / self._f_min)
        self._window = np.zeros((self._win_size, n_channels))
        self._lower_bound = self._samplerate / (2*self._max_frequency)
        order = 20
        cutoff_norm = 0.01
        self._lp_filter = signal.firwin(order - 1, cutoff_norm, window="hamming", fs=1)
        self._hp_filter_b = [1, -1]
        self._hp_filter_a = [1, -0.99]

    def process(self, buffer: np.ndarray, is_valid: bool):
        if not is_valid:
            pre_processed_win = self.pre_process(self._window)
            self._extrapolated_concealment_data = np.zeros((self._extraction_length * self._packet_size, self._n_channels))
            for n in range(self._n_channels):
                zero_crossings = self.zero_crossing_detect(pre_processed_win[:, n])
                if len(zero_crossings) < 2:
                    return np.zeros(np.shape(buffer))
                concealment_data = self.extract(self._window[:, n], zero_crossings)
                concealment_data = self.align(self._window[:,n], concealment_data)
                self._extrapolated_concealment_data[:, n] = self.extrapolate_and_fade_in(self._window[:, n], concealment_data)
            self._crossfade = True
            buffer_out = self._extrapolated_concealment_data[:self._packet_size]
        elif self._crossfade:
            buffer_out = np.zeros(np.shape(buffer))
            for n in range(self._n_channels):
                buffer_out[:, n] = self.fade_out(buffer[:, n], self._extrapolated_concealment_data[self._packet_size:, n])
            self._crossfade = False
        else:
            buffer_out = buffer

        buffer_size = np.shape(buffer_out)[0]
        self._window = np.roll(self._window, -buffer_size, axis=0)
        self._window[-buffer_size:] = buffer_out
        return buffer_out

    def pre_process(self, buffer: np.ndarray) -> np.ndarray:
        buffer = np.multiply(np.sqrt(np.abs(buffer)), np.sign(buffer))
        buffer = signal.filtfilt(self._lp_filter, [1], buffer, axis=0, padlen=self._packet_size - 1)
        buffer = signal.lfilter(self._hp_filter_b, self._hp_filter_a, buffer, axis=0)
        return buffer

    def zero_crossing_detect(self, buffer: np.ndarray) -> np.ndarray:
        zero_crossings = np.zeros(np.shape(buffer))
        zero_crossing_idx = []
        for i in range(1, len(buffer)):
            zero_crossings[i] = np.logical_xor(np.signbit(buffer[i]), np.signbit(buffer[i - 1]))
            if zero_crossings[i]: 
                if len(zero_crossing_idx) == 0 or (i - zero_crossing_idx[-1]) >= self._lower_bound:
                    zero_crossing_idx.append(i)
        return np.array(zero_crossing_idx)

    def extract(self, buffer: np.ndarray, zero_crossings: np.ndarray):
        n_periods = floor(len(zero_crossings) / 2)
        right_boundary = zero_crossings[-1]
        left_boundary = zero_crossings[-2*n_periods]
        extracted_periods = buffer[left_boundary:right_boundary]
        linear_series = np.interp(np.arange(2*self._n_m), [0, 2*self._n_m], [extracted_periods[-self._n_m + 1], extracted_periods[self._n_m]])
        extracted_periods[-self._n_m:] = linear_series[:self._n_m]
        extracted_periods[:self._n_m] = linear_series[self._n_m:2*self._n_m]
        return extracted_periods

    def align(self, buffer: np.ndarray, extracted: np.ndarray):
        end_slope = buffer[-1] - buffer[-2]
        extracted_derivative = np.diff(extracted, 1)
        zipped = list(zip(np.abs(extracted_derivative - buffer[-1]), np.abs(extracted_derivative - end_slope), range(len(extracted_derivative))))
        closest_matches = sorted(zipped, key = lambda x: x[1])[:10]
        match = sorted(closest_matches, key= lambda x: x[0])[0]
        index = match[2]
        rolled = np.roll(extracted, len(extracted) - index)
        extraction_length = self._extraction_length*self._packet_size
        extended_concealment = np.zeros(extraction_length)
        for i in range(extraction_length):
            extended_concealment[i] = rolled[i % len(rolled)]
        return extended_concealment

    def extrapolate_and_fade_in(self, buffer: np.ndarray, alligned_concealment: np.ndarray):
        end_slope = buffer[-1] - buffer[-2]
        if end_slope == 0:
            extrapolated = np.repeat(buffer[-1], self._fade_in_length - 1)
        else:
            extrapolated = np.arange(buffer[-1], self._fade_in_length*end_slope + buffer[-1], end_slope)[1:self._fade_in_length]
        window = np.arange(0, 1, 1/self._fade_in_length)[1:]
        alligned_concealment[:self._fade_in_length - 1] = np.multiply(alligned_concealment[:self._fade_in_length - 1], window) + np.multiply(extrapolated, 1 - window)
        return alligned_concealment

    def fade_out(self, buffer: np.ndarray, concealment: np.ndarray):
        fade_out_length = int(self._fade_out_length * self._packet_size)
        window = np.arange(0, 1, 1/fade_out_length)[1:]
        buffer[:fade_out_length - 1] = np.multiply(buffer[:fade_out_length - 1], window) + np.multiply(concealment[:fade_out_length - 1], 1 - window)
        return buffer


import soundfile as sf
def test_pre_process(lcc: LowCostConcealment, file: sf.SoundFile):
    audio_track = file.read()
    print(np.shape(audio_track))
    filtered_audio_track = lcc.pre_process(audio_track)
    sf.write("original_tracks/Chonks_filtered_nl.wav", filtered_audio_track, file.samplerate, file.subtype, file.endian, file.format)

def test_zero_crossing_detection(lcc: LowCostConcealment, file: sf.SoundFile, debug_print: bool = False, plot: bool = False):
    audio_track = file.read()
    filtered_audio_track = lcc.pre_process(audio_track)
    zero_crossings = lcc.zero_crossing_detect(filtered_audio_track[:, 0])
    if debug_print:
        for i in range(1, len(zero_crossings)):
            print((zero_crossings[i] - zero_crossings[i - 1]))
        print(lcc._lower_bound)
    if plot:
        n_samples = 2000
        sampled_filtered_audio = filtered_audio_track[-n_samples:]
        sampled_audio = audio_track[-n_samples:]
        offset = len(audio_track) - n_samples
        sampled_zero_crossings = zero_crossings[zero_crossings > offset] - offset
        fig = plt.figure(figsize=(12, 6), dpi=600)
        name = "Zero Crossing Detection Test"
        fig.suptitle(name)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlabel("Time [samples]")
        ax.set_ylabel("Amplitude")
        ax.set_xticks(np.arange(0, len(sampled_audio), 5))
        ax.grid(visible=True)
        ax.plot(np.arange(len(sampled_filtered_audio)), sampled_filtered_audio[:, 0], 'x', markersize=2)
        ax.plot(np.arange(len(sampled_audio)), sampled_audio[:, 0], 'x', markersize=2)
        ax.vlines(sampled_zero_crossings, -1, 1, linewidth=0.5)
        fig.savefig("original_tracks/Zero_Crossing_Test.png", bbox_inches='tight')

def test_extraction(lcc: LowCostConcealment, file: sf.SoundFile):
    n_samples = 600
    audio_track = file.read()[-2000:]
    sampled_audio = audio_track[-n_samples:]
    filtered_audio_track = lcc.pre_process(sampled_audio)
    zero_crossings = lcc.zero_crossing_detect(filtered_audio_track[:, 0])
    extracted_periods = lcc.extract(sampled_audio[:, 0], zero_crossings)
    extracted_periods = np.roll(extracted_periods, round(len(extracted_periods)/2))
    fig = plt.figure(figsize=(12, 6), dpi=600)
    name = "Extracted periods"
    fig.suptitle(name)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Time [samples]")
    ax.set_ylabel("Amplitude")
    ax.grid(visible=True)
    ax.plot(np.arange(len(extracted_periods)), extracted_periods)
    fig.savefig("original_tracks/Extracted_periods.png", bbox_inches='tight')

def test_alignment(lcc: LowCostConcealment, file: sf.SoundFile):
    n_samples = 600
    audio_track = file.read()[-2000:]
    sampled_audio = audio_track[-n_samples:]
    filtered_audio_track = lcc.pre_process(sampled_audio)
    zero_crossings = lcc.zero_crossing_detect(filtered_audio_track[:, 0])
    extracted_periods = lcc.extract(sampled_audio[:, 0], zero_crossings)
    rolled = lcc.align(audio_track[:, 0], extracted_periods)
    plot_length = len(sampled_audio) + len(rolled)
    fig = plt.figure(figsize=(12, 6), dpi=600)
    name = "Aligned concealment"
    fig.suptitle(name)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Time [samples]")
    ax.set_ylabel("Amplitude")
    ax.grid(visible=True)
    ax.plot(np.arange(len(sampled_audio)), sampled_audio[:, 0])
    ax.plot(np.arange(len(sampled_audio), plot_length), rolled)
    fig.savefig("original_tracks/Aligned_concealment.png", bbox_inches='tight')

def test_extrapolation_and_fade_in(lcc: LowCostConcealment, file: sf.SoundFile):
    n_samples = 600
    audio_track = file.read()[-2000:]
    sampled_audio = audio_track[-n_samples:]
    filtered_audio_track = lcc.pre_process(sampled_audio)
    zero_crossings = lcc.zero_crossing_detect(filtered_audio_track[:, 0])
    extracted_periods = lcc.extract(sampled_audio[:, 0], zero_crossings)
    rolled = lcc.align(audio_track[:, 0], extracted_periods)
    faded_in_concealment = lcc.extrapolate_and_fade_in(audio_track[:, 0], rolled)
    plot_length = len(sampled_audio) + len(faded_in_concealment)
    fig = plt.figure(figsize=(12, 6), dpi=600)
    name = "Extrapolation and Fade In"
    fig.suptitle(name)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Time [samples]")
    ax.set_ylabel("Amplitude")
    ax.grid(visible=True)
    ax.plot(np.arange(len(sampled_audio)), sampled_audio[:, 0])
    ax.plot(np.arange(len(sampled_audio), plot_length), faded_in_concealment)
    fig.savefig("original_tracks/Extrapolation_and_Fade_In.png", bbox_inches='tight')

def test_fade_out(lcc: LowCostConcealment, file: sf.SoundFile):
    packet_size = lcc._packet_size
    audio_track = file.read()[-2*lcc._win_size - packet_size:]
    sampled_audio = audio_track[:lcc._win_size]
    next_block = audio_track[lcc._win_size + packet_size:]
    filtered_audio_track = lcc.pre_process(sampled_audio)
    zero_crossings = lcc.zero_crossing_detect(filtered_audio_track[:, 0])
    extracted_periods = lcc.extract(sampled_audio[:, 0], zero_crossings)
    aligned = lcc.align(sampled_audio[:, 0], extracted_periods)
    faded_in_concealment = lcc.extrapolate_and_fade_in(sampled_audio[:, 0], aligned)
    faded_in_concealment_tail = faded_in_concealment[packet_size:]
    faded_out_block = lcc.fade_out(next_block[:packet_size, 0], faded_in_concealment_tail)
    plot_length = lcc._win_size + packet_size + packet_size
    fig = plt.figure(figsize=(12, 6), dpi=600)
    name = "Fade Out"
    fig.suptitle(name)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Time [samples]")
    ax.set_ylabel("Amplitude")
    ax.grid(visible=True)
    ax.plot(np.arange(len(audio_track[:plot_length])), audio_track[:plot_length, 0])
    ax.plot(np.arange(lcc._win_size, lcc._win_size + packet_size), faded_in_concealment[:packet_size])
    ax.plot(np.arange(lcc._win_size + packet_size, plot_length), faded_out_block)
    fig.savefig("original_tracks/Fade_Out.png", bbox_inches='tight')

def test_process(lcc: LowCostConcealment, file: sf.SoundFile):
    packet_size = lcc._packet_size
    n_packets = 15
    lost_packet_mask = [False, False, False, False, False, False, True, False, False, True, True, False, True, True, False]
    audio_track = file.read()[-n_packets*packet_size:]
    fig, ax = plt.subplots(lcc._n_channels, 1, sharex=True, figsize=(12, 6), dpi=600)
    name = "Process"
    fig.suptitle(name)
    ax[0].set_title("Channel " + "1" + "/" + str(lcc._n_channels))
    ax[0].set_xlabel("Time [samples]")
    ax[1].set_xlabel("Time [samples]")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    ax[0].grid(visible=True)
    ax[1].grid(visible=True)
    for packet, lost_packet in zip(range(n_packets), lost_packet_mask):
        start_idx = packet*packet_size
        end_idx = (packet+1)*packet_size
        buffer = audio_track[start_idx: end_idx]
        buffer = lcc.process(buffer, not lost_packet)
        print(start_idx)
        print(end_idx)
        print(np.shape(buffer))
        if lost_packet:
            color = 'r'
        else:
            color = 'b'
        print(color)
        for n in range(lcc._n_channels):
            ax[n].plot(np.arange(start_idx, end_idx), buffer[:, n], color)
    fig.savefig("original_tracks/Process.png", bbox_inches='tight')


# with sf.SoundFile("original_tracks/Chonks_stereo_2s.wav", "r") as file:
#     samplerate = file.samplerate
#     max_frequency = 4800
#     f_min = 80
#     beta = 1
#     n_m = 2
#     fade_in_length = 10
#     fade_out_length = 0.5
#     extraction_length = 2
#     packet_size = 64
#     n_channels = 2
#     lcc = LowCostConcealment(max_frequency, f_min, beta, n_m, fade_in_length, fade_out_length, extraction_length)
#     lcc.prepare_to_play(samplerate, packet_size, n_channels)
#     #test_zero_crossing_detection(lcc, file, plot=True)
#     #test_extraction(lcc, file)
#     #test_alignment(lcc, file)
#     #test_extrapolation_and_fade_in(lcc, file)
#     #test_fade_out(lcc, file)
#     test_process(lcc, file)
