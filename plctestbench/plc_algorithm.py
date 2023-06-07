import librosa
from math import ceil
import numpy as np
from tqdm.notebook import tqdm
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
        n_channels = np.shape(original_track)[1]
        for _ in range(n_channels - 1):
            npad.append((0, 0))
        original_track = np.pad(original_track, tuple(npad), 'constant')
        reconstructed_track = np.zeros(np.shape(original_track), np.float32)
        self.prepare_to_play(n_channels)
        j = 0

        for i in tqdm(range(n_packets), desc=self.__str__()):
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
        Placeholder function to be implemented by the derived classes.
        '''
        pass

    def tick(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        Placeholder function to be implemented by the derived classes.
        '''
        pass

    def __str__(self) -> str:
        return __class__.__name__


class ZerosPLC(PLCAlgorithm):

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

    def __str__(self) -> str:
        return __class__.__name__


class LastPacketPLC(PLCAlgorithm):

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

    def __str__(self) -> str:
        return __class__.__name__

class LowCostPLC(PLCAlgorithm):
    
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

    def __str__(self) -> str:
        return __class__.__name__


class ExternalPLC(PLCAlgorithm):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        parameters = BurgEccParameters()
        parameters.mid_filter_length = self.settings.get("mid_filter_length")
        parameters.mid_cross_fade_time = self.settings.get("mid_cross_fade_time")
        parameters.side_filter_length = self.settings.get("side_filter_length")
        parameters.side_cross_fade_time = self.settings.get("side_cross_fade_time")
        self.bec = BurgErrorConcealer(parameters)
        self.bec.set_mode(self.settings.get("ecc_mode"))
        self.bec.prepare_to_play(self.settings.get("fs"), self.settings.get("packet_size"))

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        buffer = np.transpose(buffer)
        reconstructed_buffer = np.zeros(np.shape(buffer), np.float32)
        self.bec.process(buffer, reconstructed_buffer, is_valid)
        reconstructed_buffer = np.transpose(reconstructed_buffer)
        return reconstructed_buffer

    def __str__(self) -> str:
        return __class__.__name__


class DeepLearningPLC(PLCAlgorithm):

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

    def __str__(self) -> str:
        return __class__.__name__

# class LMSRegressionPLC(PLCAlgorithm):

#     def run(self, input_wave, lost_packet_mask):
#         '''

#         '''
#         output_wave = input_wave * lost_packet_mask

#         return output_wave
