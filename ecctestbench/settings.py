

class Settings(object):

    def __init__(self, fs = 44100,
                       chans = 1,
                       buffer_size = 32,
                       N = 1024,
                       amp_scale = 1.0,
                       per = 0.0001,
                       seed = 1):
        '''
        This class containes all the settings used in the testbench.

            Input:
                fs:             sampling frequency of the tracks
                chans:          number of channels of the tracks
                buffer_size:    size of the lost packets
                N:              size of the windows used for computing
                                the output measurements
                amp_scale:      scale factor for the amplitude of the
                                tracks
                per:            Probability of Error ratio.
        '''

        self.fs = fs
        self.chans = chans
        self.buffer_size = buffer_size
        self.N = N
        self.hop = N//2
        self.amp_scale = amp_scale
        self.per = per
        self.seed = seed