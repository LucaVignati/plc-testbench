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
                       mid_filter_length: int = 2048,
                       mid_cross_fade_time: float = 0.01,
                       side_filter_length: int = 2048,
                       side_cross_fade_time: float = 0.01,
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

        # Ecc Algorithm
        self.ecc_mode = ecc_mode
        self.mid_filter_length = mid_filter_length
        self.mid_cross_fade_time = mid_cross_fade_time
        self.side_filter_length = side_filter_length
        self.side_cross_fade_time = side_cross_fade_time

        # PEAQ
        self.peaq_mode = peaq_mode

        # Plot
        self.dpi = dpi
        self.linewidth = linewidth
        self.figsize = figsize