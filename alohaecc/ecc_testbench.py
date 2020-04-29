import soundfile as sf
from .ecc_algorithm import zeros_ecc, last_packet_ecc
from .packet_drop_simulator import packet_drop_simulator
from .output_plugins import output_plugins


class ecc_testbench(object):

    '''
    The testbench class for the alohaecc algorithm.
    '''

    def __init__(self, ecc_algorithm,
                 packet_drop_simulator=packet_drop_simulator,
                 output_plugins=output_plugins,
                 buffer_size=32, fs=44100):

        """
            Initialise the parameters and testing components.

            Input:
                ecc_algorithm: The chosen method of error concealment of the \
input signal after packet loss is applied
                buffer_size: Default buffer size. Can be overriden if \
necessary.
                fs: Sample Rate. Argument can be overriden if necessary.
        """

        self.buffer_size = buffer_size
        self.ecc_algorithm = ecc_algorithm(self.buffer_size)
        self.packet_drop_simulator = packet_drop_simulator(self.buffer_size)
        self.fs = fs
        self.output_plugins = output_plugins(self.buffer_size, fs=self.fs)

    def run(self, wave_files):

        '''
            Input:
                wave_files: list of input file sources for testing

            Output:
                mse: Mean Square Error calculated for all input audio files \
and their corrected copies when packet loss simulation and error correction\
 has been applied
        '''

        self.wave_files = wave_files

        mse = []

        for wave in self.wave_files:

            input_waves, input_fs = sf.read(wave)

            assert input_fs == self.fs

            # TODO:conditional array shape checking to remove wavs in fileset \
            # with >= n channels before processing

            print("\nProcessing Audio ... %s \n" % wave)

            w_s = len(input_waves)

            lost_packet_mask = \
                self.packet_drop_simulator.generate_lost_packet_mask(w_s)
            ecc_wave = self.ecc_algorithm.run(input_waves, lost_packet_mask)
            mse.append(self.output_plugins.compute_mse(input_waves, ecc_wave))

        return mse
