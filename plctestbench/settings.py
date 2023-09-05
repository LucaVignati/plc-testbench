from plctestbench.utils import compute_hash

class Settings(object):

    def __init__(self, settings: dict=None) -> None:
        self.settings = {} if settings is None else settings.copy()

    def set_progress_monitor(self, progress_monitor):
        '''
        This method is used to set the progress monitor.

            Input:
                progress_monitor:   the progress monitor to be used.
        '''
        self.progress_monitor = progress_monitor

    def get_progress_monitor(self):
        '''
        This method is used to get the progress monitor.
        '''
        return self.progress_monitor

    def inherit_from(self, parent_settings):
        '''
        This method is used to inherit the settings from the parent node.

            Input:
                parent_settings:    the parent Settings object.
        '''
        for key, value in parent_settings.get_all().items():
            self.add(key, value)

        # Save parent hash to use in __hash__ method
        self.parent = str(hash(parent_settings))

        # Save progress_monitor
        self.set_progress_monitor(parent_settings.get_progress_monitor())

    def add(self, key, value):
        '''
        This method is used to add a setting.

            Input:
                key:    the key of the setting to be added.
                value:  the value of the setting to be added.
        '''
        if key in self.settings:
            raise KeyError(f"The key {key} is already present in the settings.")

        self.settings[key] = value

    def get(self, key):
        '''
        This method is used to retrieve the value of a setting.

            Input:
                key:    the key of the setting to be retrieved.
        '''
        if key not in self.settings:
            raise KeyError(f"The key {key} is not present in the settings.")

        return self.settings[key]

    def get_all(self):
        '''
        This method is used to retrieve all the settings.
        '''
        return self.settings

    def __hash__(self):
        '''
        This method returns the hash of the settings. It is invariant with respect
        to the order of the keys.
        '''
        string = str(self) + str(self.parent) if hasattr(self, "parent") else str(self)
        return compute_hash(string)

    def __str__(self):
        '''
        This method returns a string representation of the settings.
        '''
        string = f"{'name'}: {self.__class__.__name__}\n"
        keys = list(self.settings.keys())
        keys.sort()
        for key in keys:
            string += f"{key}: {self.settings[key]}\n"
        return string

    def __copy__(self):
        '''
        This method returns a copy of the settings.
        '''
        settings_copy = Settings(self.settings.copy())
        settings_copy.set_progress_monitor(self.get_progress_monitor())
        settings_copy.__class__ = self.__class__
        return settings_copy

class OriginalAudioSettings(Settings):

    def __init__(self, filename: str):
        '''
        This class containes the global settings.

            Input:
                fs:             sampling frequency of the track.
        '''
        super().__init__()
        self.settings["filename"] = filename

    def set_fs(self, fs):
        self.settings["fs"] = fs


class BinomialPLSSettings(Settings):

    def __init__(self, seed: int = 1,
                       packet_size: int = 32,
                       per: float = 0.0001):
        '''
        This class containes the settings for the BinomialPLS class.

            Input:
                seed:           the value used as seed for random generator functions
                per:            Probability of Error ratio.
        '''
        super().__init__()
        self.settings["seed"] = seed
        self.settings["packet_size"] = packet_size
        self.settings["per"] = per

class GilbertElliotPLSSettings(Settings):

    def __init__(self, seed: int = 1,
                       packet_size: int = 32,
                       p: float = 0.001,
                       r: float = 0.05,
                       h: float = 0.5,
                       k: float = 0.99999900):
        '''
        This class containes the settings for the GilbertElliotPLS class.

            Input:
                seed:           the value used as seed for random generator functions
                p:              p parameter of the Gilbert-Elliot model.
                r:              r parameter of the Gilbert-Elliot model.
                h:              h parameter of the Gilbert-Elliot model.
                k:              k parameter of the Gilbert-Elliot model.
        '''
        super().__init__()
        self.settings["seed"] = seed
        self.settings["packet_size"] = packet_size
        self.settings["p"] = p
        self.settings["r"] = r
        self.settings["h"] = h
        self.settings["k"] = k

class ZerosPLCSettings(Settings):

    def __init__(self):
        '''
        This class containes the settings for the ZeroPLC class.
        '''
        super().__init__()

class LastPacketPLCSettings(Settings):

    def __init__(self):
        '''
        This class containes the settings for the LastPacketPLC class.
        '''
        super().__init__()

class LowCostPLCSettings(Settings):

    def __init__(self, max_frequency: float = 4800,
                       f_min: int = 80,
                       beta: float = 1,
                       n_m: int = 2,
                       fade_in_length: int = 10,
                       fade_out_length: float = 0.5,
                       extraction_length: int = 2):
        '''
        This class containes the settings for the LowCostPLC class.

            Input:
                max_frequency:  maximum frequency of the tracks.
                f_min:          minimum frequency of the tracks.
                beta:           beta parameter of the LowCostPLC algorithm.
                n_m:            n_m parameter of the LowCostPLC algorithm.
                fade_in_length: fade_in_length parameter of the LowCostPLC algorithm.
                fade_out_length: fade_out_length parameter of the LowCostPLC algorithm.
                extraction_length: extraction_length parameter of the LowCostPLC algorithm.
        '''
        super().__init__()
        self.settings["max_frequency"] = max_frequency
        self.settings["f_min"] = f_min
        self.settings["beta"] = beta
        self.settings["n_m"] = n_m
        self.settings["fade_in_length"] = fade_in_length
        self.settings["fade_out_length"] = fade_out_length
        self.settings["extraction_length"] = extraction_length

class DeepLearningPLCSettings(Settings):

    def __init__(self, model_path: str = "dl_models/model_bs256_100epochs_0.01_1e-3_1e-7",
                       fs_dl: int = 16000,
                       context_length: int = 8,
                       hop_size: int = 160,
                       window_length: int = 160*3,
                       lower_edge_hertz: float = 40.0,
                       upper_edge_hertz: float = 7600.0,
                       num_mel_bins: int = 100):
        '''
        This class containes the settings for the DeepLearningPLC class.

            Input:
                model_path:         path to the model used for PLC.
                fs_dl:              sampling frequency of the tracks.
                context_length:     context length of the tracks.
                hop_size:           hop size of the tracks.
                window_length:      window length of the tracks.
                lower_edge_hertz:   lower edge of the tracks.
                upper_edge_hertz:   upper edge of the tracks.
                num_mel_bins:       number of mel bins of the tracks.
        '''
        super().__init__()
        self.settings["model_path"] = model_path
        self.settings["fs_dl"] = fs_dl
        self.settings["context_length_s"] = context_length
        self.settings["context_length"] = context_length * self.settings["fs_dl"]
        self.settings["hop_size"] = hop_size
        self.settings["window_length"] = window_length
        self.settings["lower_edge_hertz"] = lower_edge_hertz
        self.settings["upper_edge_hertz"] = upper_edge_hertz
        self.settings["num_mel_bins"] = num_mel_bins

class MSECalculatorSettings(Settings):

    def __init__(self,
                 N: int = 1024,
                 amp_scale: float = 1.0,):
        '''
        This class containes the settings for the MSECalculator class.

        Input:
                N:              size of the windows used for computing
                                the output measurements.
                amp_scale:      scale factor for the amplitude of the
                                tracks.
        '''
        super().__init__()
        self.settings["N"] = N
        self.settings["hop"] = N//2
        self.settings["amp_scale"] = amp_scale

class MAECalculatorSettings(Settings):

    def __init__(self,
                 N: int = 1024,
                 amp_scale: float = 1.0,):
        '''
        This class containes the settings for the MAECalculator class.

        Input:
                N:              size of the windows used for computing
                                the output measurements.
                amp_scale:      scale factor for the amplitude of the
                                tracks.
        '''
        super().__init__()
        self.settings["N"] = N
        self.settings["hop"] = N//2
        self.settings["amp_scale"] = amp_scale

class SpectralEnergyCalculatorSettings(Settings):

    def __init__(self,
                 N: int = 1024,
                 amp_scale: float = 1.0,):
        '''
        This class containes the settings for the SpectralEnergyCalculatorSettings class.

        Input:
                N:              size of the windows used for computing
                                the output measurements.
                amp_scale:      scale factor for the amplitude of the
                                tracks.
        '''
        super().__init__()
        self.settings["N"] = N
        self.settings["hop"] = N//2
        self.settings["amp_scale"] = amp_scale

class PEAQCalculatorSettings(Settings):

    def __init__(self, peaq_mode: str = 'basic'):
        '''
        This class containes the settings for the PEAQCalculator class.

            Input:
                peaq_mode:      mode of the PEAQ algorithm.
        '''
        super().__init__()
        self.settings["peaq_mode"] = peaq_mode

class PlotsSettings(Settings):

    def __init__(self, dpi: int = 300,
                       linewidth: float = 0.2,
                       figsize: int = (12, 6)):
        '''
        This class containes the settings for the Plots classes.

            Input:
                figsize:        size of the figures.
        '''
        super().__init__()
        self.settings["dpi"] = dpi
        self.settings["linewidth"] = linewidth
        self.settings["figsize"] = figsize
