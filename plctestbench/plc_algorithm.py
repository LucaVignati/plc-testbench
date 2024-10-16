from math import ceil
import librosa
import numpy as np
from burg_plc import BurgBasic
from cpp_plc_template import BasePlcTemplate
import tensorflow as tf
from plctestbench.worker import Worker
from .settings import Settings
from .low_cost_concealment import LowCostConcealment

class PLCAlgorithm(Worker):

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        
        '''
        packet_size = self.settings.get("packet_size")
        is_valid = True
        lost_packets_idx = lost_samples_idx[::packet_size]/packet_size
        track_length = len(original_track)
        n_packets = ceil(track_length/packet_size)
        rounding_difference = packet_size - track_length % packet_size
        npad = [(0, rounding_difference)]
        n_channels = np.shape(original_track)[1] if len(np.shape(original_track)) > 1 else 1
        for _ in range(n_channels - 1):
            npad.append((0, 0))
        original_track = np.pad(original_track, tuple(npad), 'constant')
        reconstructed_track = np.zeros(np.shape(original_track), np.float32)
        self.prepare_to_play(n_channels)
        j = 0

        for i in self.progress_monitor(range(n_packets), desc=str(self)):
            if i > lost_packets_idx[j] and j < len(lost_packets_idx) - 1: j += 1
            start_idx = i*packet_size
            end_idx = (i+1)*packet_size
            buffer = original_track[start_idx:end_idx]
            is_valid = not i == lost_packets_idx[j]
            reconstructed_buffer = self.tick(buffer, is_valid)
            reconstructed_track[start_idx:end_idx] = reconstructed_buffer

        return reconstructed_track[0:track_length]

    def prepare_to_play(self, n_channels):
        '''
        Not all the PLC algorithms need to prepare to play.
        This function will be executed when the PLC algorithm
        doesn't override it (because it doesn't need it) so
        this function does nothing.
        '''

    def tick(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        Placeholder function to be implemented by the derived classes.
        '''
        raise NotImplementedError

class ZerosPLC(PLCAlgorithm):
    '''
    ZerosPLC is ...
    '''    

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        Run the PLC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        PLC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''
        if is_valid:
            reconstructed_buffer = buffer
        else:
            reconstructed_buffer = np.zeros(np.shape(buffer))

        return reconstructed_buffer

class LastPacketPLC(PLCAlgorithm):
    '''
    LastPacketPLC is ...
    '''

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.previous_packet = None

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        Run the PLC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        PLC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''
        if is_valid:
            reconstructed_buffer = buffer
        else:
            if self.previous_packet is None:
                self.previous_packet = np.zeros(np.shape(buffer))
            reconstructed_buffer = self.previous_packet

        self.previous_packet = buffer

        return reconstructed_buffer

class LowCostPLC(PLCAlgorithm):
    '''
    This class implements the Low Cost Concealment (LCC) described
    in "Low-delay error concealment with low computational overhead
    for audio over ip applications" by Marco Fink and Udo Zölzer
    '''

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.lcc = LowCostConcealment(settings.get("max_frequency"),
                                      settings.get("f_min"),
                                      settings.get("beta"),
                                      settings.get("n_m"),
                                      settings.get("fade_in_length"),
                                      settings.get("fade_out_length"),
                                      settings.get("extraction_length"))
        self.samplerate = settings.get("fs")
        self.packet_size = settings.get("packet_size")

    def prepare_to_play(self, n_channels):
        self.lcc.prepare_to_play(self.samplerate, self.packet_size, n_channels)

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        return self.lcc.process(buffer, is_valid)

class BurgPLC(PLCAlgorithm):
    '''
    BurgPLC is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.train_size = settings.get("train_size")
        self.order = settings.get("order")
        self.packet_size = settings.get("packet_size")
        self.previous_valid = False
        self.coefficients = np.zeros(self.order)
        self.burg = BurgBasic(self.train_size)

    def prepare_to_play(self, n_channels):
        self.context = np.zeros((self.train_size, n_channels), np.double)

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        reconstructed_buffer = buffer
        if not is_valid:
            n_channels = np.shape(buffer)[1]
            for n_channel in range(n_channels):
                context = self.context[:, n_channel]
                if self.previous_valid:
                    self.coefficients, _ = self.burg.fit(context, self.order)
                reconstructed_buffer[:, n_channel] = self.burg.predict(context, self.coefficients, self.packet_size)

        self.context = np.roll(self.context, -self.packet_size, axis=0)
        self.context[-self.packet_size:] = reconstructed_buffer

        self.previous_valid = is_valid
        return reconstructed_buffer

class ExternalPLC(PLCAlgorithm):
    '''
    ExternalPLC is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.bpt = BasePlcTemplate()
        self.bpt.prepare_to_play(self.settings.get("fs"), self.settings.get("packet_size"))

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        buffer = np.transpose(buffer)
        reconstructed_buffer = np.zeros(np.shape(buffer), np.float32)
        self.bpt.process(buffer, reconstructed_buffer, is_valid)
        reconstructed_buffer = np.transpose(reconstructed_buffer)
        return reconstructed_buffer

class DeepLearningPLC(PLCAlgorithm):
    '''
    DeepLearningPLC is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = tf.keras.models.load_model(settings.get("model_path"), compile=False)
        self.fs_dl = settings.get("fs_dl")
        self.context_size_s = settings.get("context_length_s")
        self.context_size = settings.get("context_length")
        self.hop_size = settings.get("hop_size")
        self.window_length = settings.get("window_length")
        self.lower_edge_hertz = settings.get("lower_edge_hertz")
        self.upper_edge_hertz = settings.get("upper_edge_hertz")
        self.num_mel_bins = settings.get("num_mel_bins")
        self.sample_rate = settings.get("fs")
        self.packet_size = settings.get("packet_size")

    def prepare_to_play(self, n_channels):
        self.context = np.zeros((self.context_size_s*self.sample_rate, n_channels))

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        if is_valid:
            reconstructed_buffer = buffer
        else:
            reconstructed_buffer = self.predict_reconstructed_buffer(buffer)

        self.context = np.roll(self.context, -self.packet_size, axis=1)
        self.context[-self.packet_size:, :] = reconstructed_buffer
        return reconstructed_buffer

    def compute_spectrogram(self, context, fs):
        return librosa.feature.melspectrogram(y=np.pad(context, (0, self.window_length-self.hop_size)), sr=fs, n_fft=self.window_length, hop_length=self.hop_size, win_length=self.window_length,
    center=False, n_mels=self.num_mel_bins, fmin=self.lower_edge_hertz, fmax=self.upper_edge_hertz)

    def predict_reconstructed_buffer(self, buffer):
        reconstructed_buffer = np.zeros(np.shape(buffer.T))
        context = librosa.resample(self.context.T, orig_sr=self.sample_rate, target_sr=self.fs_dl).T
        for channel_index in range(np.shape(buffer)[1]):
            spectrogram_2s = self.compute_spectrogram(context[-round(self.context_size/4):, channel_index], self.fs_dl)
            #spectrogram_4s = self.compute_spectrogram(librosa.resample(context[-round(self.context_size/2):, channel_index], orig_sr=self.fs_dl, target_sr=self.fs_dl/2), self.fs_dl/2)
            #spectrogram_8s = self.compute_spectrogram(librosa.resample(context[:, channel_index], orig_sr=self.fs_dl, target_sr=self.fs_dl/4), self.fs_dl/4)
            spectrograms = np.expand_dims(spectrogram_2s, axis=0)
            last_packet = np.expand_dims(context[-self.packet_size:, channel_index], axis=0)
            reconstructed_buffer[channel_index, :] = self.model((spectrograms, last_packet))
        return reconstructed_buffer.T
