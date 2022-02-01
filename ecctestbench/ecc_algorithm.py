from math import ceil
import numpy as np
from tqdm.notebook import tqdm, trange
from ecctestbench.worker import Worker
from .settings import Settings
from .low_cost_concealment import LowCostConcealment
from ecc_external import BurgErrorConcealer, BurgEccParameters

class ECCAlgorithm(Worker):

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        
        '''
        packet_size = self.settings.packet_size
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
        ecc_track = np.zeros(np.shape(original_track), np.float32)
        self.prepare_to_play(n_channels)
        j = 0

        for i in tqdm(range(n_packets), desc=self.__str__()):
            if i > lost_packets_idx[j] and j < len(lost_packets_idx) - 1: j += 1
            start_idx = i*packet_size
            end_idx = (i+1)*packet_size
            buffer = original_track[start_idx:end_idx]
            is_valid = not i == lost_packets_idx[j]
            buffer_ecc = self.tick(buffer, is_valid)
            ecc_track[start_idx:end_idx] = buffer_ecc

        return ecc_track[0:track_length]

    def prepare_to_play(self, n_channels):
        pass

    def tick(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        '''
        
        '''
        pass

    def __str__(self) -> str:
        return __class__.__name__


class ZerosEcc(ECCAlgorithm):

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        Run the ECC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        ECC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''
        if is_valid:
            ecc_buffer = buffer
        else:
            ecc_buffer = np.zeros(np.shape(buffer))

        return ecc_buffer

    def __str__(self) -> str:
        return __class__.__name__


class LastPacketEcc(ECCAlgorithm):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.previous_packet = None

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        Run the ECC algorithm on the input_wave signal and
        generate an output signal of the same length using the
        ECC algorithm.

            Input:
                input_wave       : length-N numpy array
                lost_packet_mask : length-N numpy array where values are
                either 1 (valid sample) or 0 (dropped sample)

            Output:
                output_wave: length-N error corrected numpy array
        '''
        if is_valid:
            ecc_buffer = buffer
        else:
            if self.previous_packet is None:
                self.previous_packet = np.zeros(np.shape(buffer))
            ecc_buffer = self.previous_packet
        
        self.previous_packet = buffer
            
        return ecc_buffer

    def __str__(self) -> str:
        return __class__.__name__

class LowCostEcc(ECCAlgorithm):
    
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.lcc = LowCostConcealment(settings.max_frequency,
                                      settings.f_min,
                                      settings.beta,
                                      settings.n_m,
                                      settings.fade_in_length,
                                      settings.fade_out_length,
                                      settings.extraction_length)
        self.samplerate = settings.fs
        self.packet_size = settings.packet_size

    def prepare_to_play(self, n_channels):
        self.lcc.prepare_to_play(self.samplerate, self.packet_size, n_channels)

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        return self.lcc.process(buffer, is_valid)

    def __str__(self) -> str:
        return __class__.__name__


class ExternalEcc(ECCAlgorithm):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        parameters = BurgEccParameters()
        parameters.mid_filter_length = self.settings.mid_filter_length
        parameters.mid_cross_fade_time = self.settings.mid_cross_fade_time
        parameters.side_filter_length = self.settings.side_filter_length
        parameters.side_cross_fade_time = self.settings.side_cross_fade_time
        self.bec = BurgErrorConcealer(parameters)
        self.bec.set_mode(self.settings.ecc_mode)
        self.bec.prepare_to_play(self.settings.fs, self.settings.packet_size)

    def tick(self, buffer: np.ndarray, is_valid: bool):
        '''
        
        '''
        buffer = np.transpose(buffer)
        ecc_buffer = np.zeros(np.shape(buffer), np.float32)
        self.bec.process(buffer, ecc_buffer, is_valid)
        ecc_buffer = np.transpose(ecc_buffer)
        return ecc_buffer

    def __str__(self) -> str:
        return __class__.__name__


# class LMSRegressionECC(ECCAlgorithm):

#     def run(self, input_wave, lost_packet_mask):
#         '''

#         '''
#         output_wave = input_wave * lost_packet_mask

#         return output_wave
