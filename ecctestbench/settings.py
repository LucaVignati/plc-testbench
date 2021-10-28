import numpy.random as npr

class Settings(object):

    def __init__(self, fs = 44100,
                       chans = 1,
                       packet_size = 32,
                       N = 1024,
                       amp_scale = 1.0,
                       seed = 1,
                       per = 0.0001,
                       p = 0.0001,
                       r = 0.05,
                       h = 0.5,
                       k = 0.99999900,
                       dpi = 300,
                       linewidth = 0.2,
                       figsize = (12, 6)):
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

        # Plot
        self.dpi = dpi
        self.linewidth = linewidth
        self.figsize = figsize