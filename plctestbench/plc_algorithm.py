from math import ceil
import librosa
import numpy as np
from burg_plc import BurgBasic
from cpp_plc_template import BasePlcTemplate
import tensorflow as tf
from plctestbench.worker import Worker
from .settings import Settings
from .low_cost_concealment import LowCostConcealment
from .crossfade import Crossfade

class PLCAlgorithm(Worker):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.packet_size = self.settings.get("packet_size")
        self.crossfade = Crossfade(self.settings, self.settings.get("crossfade"))
        if self.settings.get("fade_in").get("length") > self.packet_size:
            raise ValueError("fade in length cannot be longer than the packet size")
        self.fade_in = Crossfade(self.settings, self.settings.get("fade_in"))
        try:
            self.context_length = self.settings.get("context_length") * self.settings.get("fs") / 1000
        except:
            self.context_length = self.packet_size

    

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        
        '''
        def zero_pad(original_track: np.ndarray):
            '''
            '''
            rounding_difference = self.packet_size - track_length % self.packet_size
            npad = [(0, rounding_difference)]
            for _ in range(self.n_channels - 1):
                npad.append((0, 0))
            return np.pad(original_track, tuple(npad), 'constant')
        
        self.n_channels = np.shape(original_track)[1] if len(np.shape(original_track)) > 1 else 1
        lost_packets_idx = lost_samples_idx[::self.packet_size]/self.packet_size
        track_length = len(original_track)
        n_packets = ceil(track_length/self.packet_size)
        original_track = zero_pad(original_track)
        reconstructed_track = np.zeros(np.shape(original_track), np.float32)
        self._prepare_to_play()
        
        j = 0
        for i in self.progress_monitor(range(n_packets), desc=str(self)):
            if i > lost_packets_idx[j] and j < len(lost_packets_idx) - 1: j += 1
            start_idx = i*self.packet_size
            end_idx = (i+1)*self.packet_size
            buffer = original_track[start_idx:end_idx]
            is_valid = not i == lost_packets_idx[j]
            reconstructed_buffer = self._tick(buffer, is_valid)
            reconstructed_track[start_idx:end_idx] = reconstructed_buffer

        return reconstructed_track[:track_length]

    def _prepare_to_play(self):
        '''
        Not all the PLC algorithms need to prepare to play.
        This function will be executed when the PLC algorithm
        doesn't override it (because it doesn't need it) so
        this function does nothing.
        '''
        self.context = np.zeros((self.context_length, self.n_channels))

    def _tick(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        This function is called for every buffer. It manages the creation
        of the predicted buffer and the crossfading with the following audio.
        '''
        output_buffer = self._a_priori(buffer, is_valid)
        if is_valid:
            output_buffer = self._crossfade(output_buffer)
        else:
            output_buffer = self._predict(output_buffer)
            output_buffer = self._fade_in(output_buffer)
        output_buffer = self._a_posteriori(output_buffer, is_valid)
        return output_buffer

    def _a_priori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        This function is called for every buffer.
        '''
        return buffer

    def _a_posteriori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        This function is called for every buffer.
        '''
        self.context = np.roll(self.context, -self.packet_size, axis=1)
        self.context[-self.packet_size:, :] = buffer
        return buffer

    def _fade_in(self, buffer: np.ndarray) -> np.ndarray:
        '''
        This function is called for every buffer.
        '''
        output_buffer = self.fade_in(self.context[-self.packet_size:], buffer)
        return output_buffer
    
    def _predict(self, buffer: np.ndarray) -> np.ndarray:
        '''
        This function is called for every buffer.
        '''
        return buffer

    def _crossfade(self, buffer: np.ndarray) -> np.ndarray:
        '''
        This function is called for every buffer.
        '''
        output_buffer = buffer
        if self.crossfade.ongoing():
            prediction = self._predict(buffer)
            output_buffer = self.crossfade(prediction, buffer)
        return output_buffer

class ZerosPLC(PLCAlgorithm):
    '''
    ZerosPLC is ...
    '''
    
    def _predict(self, buffer: np.ndarray):
        '''
        
        '''
        return np.zeros(np.shape(buffer))

class LastPacketPLC(PLCAlgorithm):
    '''
    LastPacketPLC is ...
    '''

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.mirror_x = settings.get("mirror_x")
        self.mirror_y = settings.get("mirror_y")
        self.clip_strategy = settings.get("clip_strategy")

    def _predict(self, _: np.ndarray):
        '''
        
        '''
        def _flip_in_place(buffer: np.ndarray):
            '''
            '''
            return -(buffer - buffer[0]) + buffer[0]

        reconstructed_buffer = self.context[-self.packet_size:]
        if self.mirror_x:
            reconstructed_buffer = np.flip(reconstructed_buffer, axis=0)
            if self.mirror_y:
                for channel in range(self.n_channels):
                    reconstructed_buffer[:, channel] = _flip_in_place(reconstructed_buffer[:, channel])
                    for sample in range(np.shape(reconstructed_buffer)[0]):
                        if abs(sample) > 1:
                            if self.clip_strategy == "subtract":
                                reconstructed_buffer[sample:, channel] = reconstructed_buffer[sample:, channel] - (sample - np.sign(sample))
                            elif self.clip_strategy == "flip":
                                reconstructed_buffer[sample:, channel] = _flip_in_place(reconstructed_buffer[sample:, channel])
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

    def _prepare_to_play(self):
        super()._prepare_to_play()
        self.lcc.prepare_to_play(self.samplerate, self.packet_size, self.n_channels)

    def _a_priori(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        return self.lcc.process(buffer, is_valid)

class BurgPLC(PLCAlgorithm):
    '''
    BurgPLC is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.order = settings.get("order")
        self.previous_valid = False
        self.coefficients = np.zeros(self.order)
        context_length_samples = round(self.context_length/1000*self.settings.get("fs"))
        self.burg = BurgBasic(context_length_samples)

    def _predict(self, buffer: np.ndarray):
        '''
        '''
        reconstructed_buffer = np.zeros(np.shape(buffer), np.float32)
        n_channels = np.shape(buffer)[1]
        for n_channel in range(n_channels):
            context = self.context[:, n_channel]
            if self.previous_valid:
                self.coefficients, _ = self.burg.fit(context, self.order)
            reconstructed_buffer[:, n_channel] = self.burg.predict(context, self.coefficients, self.packet_size)
        return reconstructed_buffer

    def _a_posteriori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        '''
        super()._a_posteriori(buffer, is_valid)
        self.previous_valid = is_valid
        return buffer

class ExternalPLC(PLCAlgorithm):
    '''
    ExternalPLC is ...
    '''
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.bpt = BasePlcTemplate()
        self.bpt.prepare_to_play(self.settings.get("fs"), self.packet_size)

    def _tick(self, buffer: np.ndarray, is_valid: bool):
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
        self.context_length = settings.get("context_length")
        self.context_length_samples = settings.get("context_length_samples")
        self.hop_size = settings.get("hop_size")
        self.window_length = settings.get("window_length")
        self.lower_edge_hertz = settings.get("lower_edge_hertz")
        self.upper_edge_hertz = settings.get("upper_edge_hertz")
        self.num_mel_bins = settings.get("num_mel_bins")
        self.sample_rate = settings.get("fs")

    def _predict(self, buffer: np.ndarray):
        '''
        '''
        return self._predict_reconstructed_buffer(buffer)

    def _compute_spectrogram(self, context, fs):
        return librosa.feature.melspectrogram(y=np.pad(context, (0, self.window_length-self.hop_size)), sr=fs, n_fft=self.window_length, hop_length=self.hop_size, win_length=self.window_length,
    center=False, n_mels=self.num_mel_bins, fmin=self.lower_edge_hertz, fmax=self.upper_edge_hertz)

    def _predict_reconstructed_buffer(self, buffer):
        reconstructed_buffer = np.zeros(np.shape(buffer.T))
        context = librosa.resample(self.context.T, orig_sr=self.sample_rate, target_sr=self.fs_dl).T
        for channel_index in range(np.shape(buffer)[1]):
            spectrogram_2s = self._compute_spectrogram(context[-round(self.context_length_samples/4):, channel_index], self.fs_dl)
            #spectrogram_4s = self._compute_spectrogram(librosa.resample(context[-round(self.context_length_samples/2):, channel_index], orig_sr=self.fs_dl, target_sr=self.fs_dl/2), self.fs_dl/2)
            #spectrogram_8s = self._compute_spectrogram(librosa.resample(context[:, channel_index], orig_sr=self.fs_dl, target_sr=self.fs_dl/4), self.fs_dl/4)
            spectrograms = np.expand_dims(spectrogram_2s, axis=0)
            last_packet = np.expand_dims(context[-self.packet_size:, channel_index], axis=0)
            reconstructed_buffer[channel_index, :] = self.model((spectrograms, last_packet))
        return reconstructed_buffer.T
