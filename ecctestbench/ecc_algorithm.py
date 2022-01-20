import numpy as np
from ecctestbench.worker import Worker
from .settings import Settings
from ecc_external import BurgErrorConcealer, BurgEccParameters

class ECCAlgorithm(Worker):

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray):
        '''
        
        '''
        packet_size = self.settings.packet_size
        is_valid = True
        lost_packets_idx = lost_samples_idx[::packet_size]
        track_length = len(original_track)
        n_packets = int(track_length/packet_size)
        rounding_difference = packet_size - track_length % packet_size
        np.pad(original_track, (0, rounding_difference), 'constant')
        ecc_track = np.zeros(np.shape(original_track), np.float32)

        for i in range(n_packets):
            start_idx = i*packet_size
            end_idx = (i+1)*packet_size
            buffer = original_track[start_idx:end_idx]
            is_valid = not (i in lost_packets_idx)
            buffer_ecc = self.tick(buffer, is_valid)
            ecc_track[start_idx:end_idx] = buffer_ecc

        return ecc_track[0:track_length]


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
        #print(buffer.ndim)
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
