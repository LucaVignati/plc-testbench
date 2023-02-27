import numpy.random as npr
from ecc_external import EccMode

class Settings(object):

    def __init__(self, fs: int = 44100,
                       chans: int = 1,
                       packet_size: int = 32,
                       N: int = 1024,
                       amp_scale: float = 1.0,
                       seed: int = 1,
                       per: float = 0.0001,
                       p: float = 0.001,
                       r: float = 0.05,
                       h: float = 0.5,
                       k: float = 0.99999900,
                       ecc_mode: EccMode = EccMode.STEREO,
                       mid_filter_length: int = 256,
                       mid_cross_fade_time: float = 0.025,
                       side_filter_length: int = 32,
                       side_cross_fade_time: float = 0.025,
                       max_frequency: float = 4800,
                       f_min: int = 80,
                       beta: float = 1,
                       n_m: int = 2,
                       fade_in_length: int = 10,
                       fade_out_length: float = 0.5,
                       extraction_length: int = 2,
                       model_path: str = '',
                       fs_dl: int = 16000,
                       context_length: int = 8,
                       hop_size: int = 160,
                       window_length: int = 160*3,
                       lower_edge_hertz: float = 40.0,
                       upper_edge_hertz: float = 7600.0,
                       num_mel_bins: int = 100,
                       peaq_mode: str = "basic",
                       dpi: int = 300,
                       linewidth: float = 0.2,
                       figsize: int = (12, 6)):
        '''
        This class containes all the settings used in the testbench.

            Input:
                fs:             sampling frequency of the tracks.
                chans:          number of channels of the tracks.
                packet_size:    size of audio packets in samples.
                N:              size of the windows used for computing
                                the output measurements.
                amp_scale:      scale factor for the amplitude of the
                                tracks.
                seed:           the value used as seed for random generator functions
                per:            Probability of Error ratio.
                p:              p parameter of the Gilbert-Elliot model.
                r:              r parameter of the Gilbert-Elliot model.
                h:              h parameter of the Gilbert-Elliot model.
                k:              k parameter of the Gilbert-Elliot model.
                dpi:            pixel density of plots.
                linewidth:      width of lines in plots.
                figsize:        size of plots.
        '''

        self.fs = fs
        self.chans = chans
        self.packet_size = packet_size
        self.N = N
        self.hop = N//2
        self.amp_scale = amp_scale

        # Loss Simulator
        self.seed = seed
        self.per = per
        self.p = p
        self.r = r
        self.h = h
        self.k = k

        ## Ecc Algorithm
        # EccExternal
        self.ecc_mode = ecc_mode
        self.mid_filter_length = mid_filter_length
        self.mid_cross_fade_time = mid_cross_fade_time
        self.side_filter_length = side_filter_length
        self.side_cross_fade_time = side_cross_fade_time
        # LowCostECC
        self.max_frequency = max_frequency
        self.f_min = f_min
        self.beta = beta
        self.n_m = n_m
        self.fade_in_length = fade_in_length
        self.fade_out_length = fade_out_length
        self.extraction_length = extraction_length
        # DeepLearningECC
        self.model_path = model_path
        self.fs_dl = fs_dl
        self.context_length_s = context_length
        self.context_length = context_length * self.fs_dl
        self.hop_size = hop_size
        self.window_length = window_length
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins

        # PEAQ
        self.peaq_mode = peaq_mode

        # Plot
        self.dpi = dpi
        self.linewidth = linewidth
        self.figsize = figsize