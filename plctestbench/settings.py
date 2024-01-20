from enum import Enum
from typing import List
from typing_inspect import get_parameters

from plctestbench.utils import compute_hash, get_class

class Settings(object):

    def __init__(self, settings: dict=None) -> None:
        self.settings = {} if settings is None else settings.copy()
        self.unflatten()

    def unflatten(self):
        for key, value in list(self.settings.items())  :
            if '-' in key:
                original_key = str(key)
                key, cls = key.split('-')
                obj = Settings(value) if Settings in get_class(cls).__mro__ else get_class(cls)(value)
                obj.__class__ = get_class(cls)
                if '~' in key:
                    key, _ = key.split('~')
                    self.settings.setdefault(key, []).append(obj)
                else:
                    self.settings[key] = obj
                del self.settings[original_key]

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
            if key != self:
                if isinstance(value, Settings):
                    value.unflatten()
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
        return self.to_dict()

    def to_dict(self):
        '''
        This method is used to convert the settings to a dictionary.
        '''
        def parse_values(key, value, to_delete: list = [], to_add: dict = {}):
            if isinstance(value, Settings):
                to_add[key + '-' + value.__class__.__name__] = value.get_all()
                to_delete.append(key)
            if isinstance(value, Enum):
                to_add[key + '-' + value.__class__.__name__] = value.name
                to_delete.append(key)
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], Settings):
                    for idx, item in enumerate(value):
                        _, new_dict_entry = parse_values(key + '~' + str(idx), item)
                        if len(new_dict_entry) > 0:
                            to_add.update(new_dict_entry)
                    if len(new_dict_entry) > 0:
                        to_delete.append(key)

            return to_delete, to_add

        to_delete = []
        to_add = {}
        for key, value in self.settings.items():
            parse_values(key, value, to_delete, to_add)
        for key in to_delete:
            del self.settings[key]
        self.settings.update(to_add)
        return self.settings

    def __hash__(self):
        '''
        This method returns the hash of the settings. It is invariant with respect
        to the order of the keys.
        '''
        string = str(self) + str(self.parent) if hasattr(self, "parent") else str(self)
        return compute_hash(string)

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
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

class MultibandCrossfadeSettings(Settings):

    def __init__(self, frequencies: list = [],
                       order: int = 4):
        '''
        This class containes the settings for the MultibandCrossfade class.

            Input:
                crossfade_settings:     list of the settings for the crossfades.
        '''
        super().__init__()
        self.settings["frequencies"] = frequencies
        self.settings["order"] = order

class CrossfadeSettings(Settings):

    def __init__(self) -> None:
        super().__init__()
        self.length_in_samples = 0
        self.settings["length_in_samples"] = 0

class CrossfadeFunction(Enum):
    power = "power"
    sinusoidal = "sinusoidal"
    
class CrossfadeType(Enum):
    power = "power"
    amplitude = "amplitude"
        
class NoCrossfadeSettings(CrossfadeSettings):

    def __init__(self):
        '''
        This class containes the settings for the NoCrossfade class.
        '''
        super().__init__()
        self.settings["length"] = 0
        self.settings["function"] = CrossfadeFunction.power
        self.settings["exponent"] = 1.0
        self.settings["type"] = CrossfadeType.power
        
class ManualCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = 10,
                       function: CrossfadeFunction = CrossfadeFunction.power,
                       type: CrossfadeType = CrossfadeType.power,
                       exponent: float = 1.0):
        '''
        This class containes the settings for the PowerCrossfade class.

            Input:
                length:     length of the crossfade.
                funtion:    function used for the crossfade.
                type:       type of the crossfade.
                exponent:   exponent of the crossfade.
        '''
        super().__init__()
        self.settings["length"] = length
        self.settings["function"] = CrossfadeFunction.power
        if function == CrossfadeFunction.power:
            self.settings["exponent"] = exponent
        self.settings["type"] = type

class LinearCrossfadeSettings(CrossfadeSettings):
    
    def __init__(self, length: int = 10, type: CrossfadeType = CrossfadeType.power):
        '''
        This class containes the settings for the LinearCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__()
        self.settings["length"] = length
        self.settings["function"] = CrossfadeFunction.power
        self.settings["exponent"] = 1.0
        self.settings["type"] = type

class QuadraticCrossfadeSettings(CrossfadeSettings):
    
    def __init__(self, length: int = 10, type: CrossfadeType = CrossfadeType.power):
        '''
        This class containes the settings for the QuadraticCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__()
        self.settings["length"] = length
        self.settings["function"] = CrossfadeFunction.power
        self.settings["exponent"] = 2.0
        self.settings["type"] = type

class CubicCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = 10, type: CrossfadeType = CrossfadeType.power):
        '''
        This class containes the settings for the CubicCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__()
        self.settings["length"] = length
        self.settings["function"] = CrossfadeFunction.power
        self.settings["exponent"] = 3.0
        self.settings["type"] = type

class SinusoidalCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = 10, type: CrossfadeType = CrossfadeType.power):
        '''
        This class containes the settings for the SinusoidalCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__()
        self.settings["length"] = length
        self.settings["function"] = CrossfadeFunction.sinusoidal
        self.settings["type"] = type

class PLCSettings(Settings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       frequencies: List[int] = [],
                       order: int = 4) -> None:
        super().__init__()
        self.settings["crossfade"] = crossfade if crossfade is not None else [NoCrossfadeSettings() for i in range(len(frequencies) + 1)]
        self.settings["fade_in"] = fade_in if fade_in is not None else [NoCrossfadeSettings()]
        self.settings["frequencies"] = frequencies
        self.settings["order"] = order
        
        #assert len(self.settings["crossfade"]) == len(self.settings["frequencies"]), "The number of crossfade settings must be equal to the number of frequencies."

class ZerosPLCSettings(PLCSettings):               

    def __init__(sel, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                      fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                      frequencies: List[int] = []) -> None:
        '''
        This class containes the settings for the ZeroPLC class.
        '''
        super().__init__(crossfade, fade_in, frequencies)

class LastPacketPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       mirror_x: bool = False,
                       mirror_y: bool = False,
                       clip_strategy: str = "subtract",
                       frequencies: List[int] = []):
        '''
        This class containes the settings for the LastPacketPLC class.
        '''
        super().__init__(crossfade, fade_in, frequencies)
        self.settings["mirror_x"] = mirror_x
        self.settings["mirror_y"] = mirror_y
        self.settings["clip_strategy"] = clip_strategy


class LowCostPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       max_frequency: float = 4800,
                       f_min: int = 80,
                       beta: float = 1,
                       n_m: int = 2,
                       fade_in_length: int = 10,
                       fade_out_length: float = 0.5,
                       extraction_length: int = 2,
                       frequencies: List[int] = []):
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
        super().__init__(crossfade, fade_in, frequencies)
        self.settings["max_frequency"] = max_frequency
        self.settings["f_min"] = f_min
        self.settings["beta"] = beta
        self.settings["n_m"] = n_m
        self.settings["fade_in_length"] = fade_in_length
        self.settings["fade_out_length"] = fade_out_length
        self.settings["extraction_length"] = extraction_length

class BurgPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       context_length: int = 100,
                       order: int = 1,
                       frequencies: List[int] = []):
        '''
        This class containes the settings for the BurgPLC class.

            Input:
                context_length: size of the training set.
                order:          order of the Burg algorithm.
        '''
        super().__init__(crossfade, fade_in, frequencies)
        self.settings["context_length"] = context_length
        self.settings["order"] = order

class ExternalPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       frequencies: List[int] = []):
        '''
        This class containes the settings for the ExternalPLC class.
        '''
        super().__init__(crossfade, fade_in, frequencies)

class DeepLearningPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       fade_in: List[CrossfadeSettings] = [NoCrossfadeSettings()],
                       model_path: str = "dl_models/model_bs256_100epochs_0.01_1e-3_1e-7",
                       fs_dl: int = 16000,
                       context_length: int = 8000,
                       hop_size: int = 160,
                       window_length: int = 160*3,
                       lower_edge_hertz: float = 40.0,
                       upper_edge_hertz: float = 7600.0,
                       num_mel_bins: int = 100,
                       frequencies: List[int] = []):
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
        super().__init__(crossfade, fade_in, frequencies)
        self.settings["model_path"] = model_path
        self.settings["fs_dl"] = fs_dl
        self.settings["context_length"] = context_length
        self.settings["context_length_samples"] = context_length / 1000.0 * self.settings["fs_dl"]
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

class PEAQMode(Enum):
    basic = 1
    advanced = 2

class PEAQCalculatorSettings(Settings):

    def __init__(self, peaq_mode: PEAQMode = PEAQMode.basic):
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
