from enum import Enum
from typing_inspect import get_parameters
from inspect import isclass
from typing import List, Dict
from copy import deepcopy

from plctestbench.utils import compute_hash, get_class, relative_to_root

class Settings(object):

    def __init__(self, settings: dict=None) -> None:
        self.settings = {} if settings is None else self.from_dict(settings)

    def from_dict(self, settings_dict):
        '''
        This method is used to convert a dictionary back to settings.
        '''
        def reconstruct_values(key, value):
            special_chars = list(filter(lambda x: x in ['-', '~', '&', '$', '#'], list(str(key))))
            last_special_char = special_chars[-1] if len(special_chars) > 0 else None
            if last_special_char == '-':
                key, class_name = key.split('-')
                new_value = value
                clazz = get_class(class_name)
                if Settings in clazz.__mro__:
                    new_value = clazz()
                    new_value.settings = self.from_dict(value)
                else:
                    new_value = get_class(class_name)(value)
                return key, new_value
            elif last_special_char == '~':
                key, idx = key.split('~')
                return key, [value]
            elif last_special_char == '&':
                key, idx = key.split('&')
                return key, (value)
            elif last_special_char == '$':
                key, subkey = key.split('$')
                return key, {subkey: value}
            elif last_special_char == '#':
                key = key.rstrip('#')
                return key, globals()[value]
            else:
                return key, value

        new_settings = {}
        for key, value in settings_dict.items():
            new_key, new_value = reconstruct_values(key, value)
            if new_key in new_settings and isinstance(new_settings[new_key], list):
                new_settings[new_key] += new_value
            elif new_key in new_settings and isinstance(new_settings[new_key], tuple):
                new_settings[new_key] = new_settings[new_key] + (new_value,)
            elif new_key in new_settings and isinstance(new_settings[new_key], dict):
                new_settings[new_key].update(new_value)
            else:
                new_settings[new_key] = new_value

        return new_settings if new_settings.keys() == settings_dict.keys() else self.from_dict(new_settings)

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

    def to_dict(self):
        '''
        This method is used to convert the settings to a dictionary.
        ''' 
        def parse_values(key, value, to_delete: list = [], to_add: dict = {}):
            if isinstance(value, Settings):
                to_add[key + '-' + value.__class__.__name__] = value.to_dict()
                to_delete.append(key)
            if isinstance(value, Enum):
                to_add[key + '-' + value.__class__.__name__] = value.name
                to_delete.append(key)
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    _, new_dict_entry = parse_values(key + '~' + str(idx), item, [], {})
                    if len(new_dict_entry) > 0:
                        to_add.update(new_dict_entry)
                if 'new_dict_entry' in locals() and len(new_dict_entry) > 0:
                    to_delete.append(key)
            if isinstance(value, tuple):
                for idx, item in enumerate(value):
                    _, new_dict_entry = parse_values(key + '&' + str(idx), item, [], {})
                    if len(new_dict_entry) > 0:
                        to_add.update(new_dict_entry)
                if 'new_dict_entry' in locals() and len(new_dict_entry) > 0:
                    to_delete.append(key)
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    _, new_dict_entry = parse_values(key + '$' + subkey, subvalue, [], {})
                    if len(new_dict_entry) > 0:
                        to_add.update(new_dict_entry)
                if 'new_dict_entry' in locals() and len(new_dict_entry) > 0:
                    to_delete.append(key)
            if isclass(value):
                to_add[key + '#'] = value.__name__
                to_delete.append(key)
            return to_delete, to_add

        to_delete = []
        to_add = {}
        settings_dict = self.settings.copy()
        for key, value in self.settings.items():
            parse_values(key, value, to_delete, to_add)
        for key in to_delete:
            del settings_dict[key]
        settings_dict.update(to_add)
        return settings_dict
    
    def clone(self):
        return deepcopy(self)
    
    def assert_setting_is_number(self, setting_name: str):
        value = self.get(setting_name)
        if not isinstance(value, (int, float)):
            raise AssertionError(f"{setting_name} is not a number")
    
    def assert_setting_is_number_greater_than(self, setting_name: str, min_value):
        self.assert_setting_is_number(setting_name)
        value = self.get(setting_name)
        if value < min_value:
            raise AssertionError(f"{setting_name} should be greater or equal than {min_value}")
    
    def assert_setting_is_number_less_than(self, setting_name: str, max_value):
        self.assert_setting_is_number(setting_name)
        value = self.get(setting_name)
        if value > max_value:
            raise AssertionError(f"{setting_name} should be smaller or equal than {max_value}")

    def assert_setting_is_number_in_range(self, setting_name: str, min_value = None, max_value = None):
        try:
            if min_value is not None:
                self.assert_setting_is_number_greater_than(setting_name, min_value)
            if max_value is not None:
                self.assert_setting_is_number_less_than(setting_name, max_value)
        except AssertionError as e:
            if min_value is not None and max_value is not None:
                raise AssertionError(f"{setting_name} should be in the range [{min_value}, {max_value}]")
            else:
                raise e
    
    def __change_setting__(self, name: str, value, change_callback: callable = None):
        if value == self.get(name):
            return self
        
        cloned_settings = self.clone()
        cloned_settings.settings[name] = value
        
        if (change_callback and callable(change_callback)):
            change_callback(cloned_settings)
        
        cloned_settings.__validate__()
        return cloned_settings
    
    def __validate__(self):
        pass

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
        settings_dict = self.to_dict()
        string = f"{'name'}: {self.__class__.__name__}\n"
        keys = list(settings_dict.keys())
        keys.sort()
        for key in keys:
            string += f"{key}: {settings_dict[key]}\n"
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

        self.__validate__()

    def __validate__(self):
        self.assert_setting_is_number_in_range("packet_size", min_value=1)
        self.assert_setting_is_number_in_range("per", 0, 1)

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

        self.__validate__()
    
    def __validate__(self):
        self.assert_setting_is_number_in_range("packet_size", min_value=1)
        self.assert_setting_is_number_in_range("p", 0, 1)
        self.assert_setting_is_number_in_range("r", 0, 1)
        self.assert_setting_is_number_in_range("h", 0, 1)
        self.assert_setting_is_number_in_range("k", 0, 1)

class StereoImageType(Enum):
    dual_mono = "dual_mono"
    mid_side = "mid_side"

class CrossfadeFunction(Enum):
    power = "power"
    sinusoidal = "sinusoidal"
    
    def toJson(self):
        return self.value
    
class CrossfadeType(Enum):
    power = "power"
    amplitude = "amplitude"
    
    def toJson(self):
        return self.value

class CrossfadeSettings(Settings):

    def __init__(self, length: int, function: CrossfadeFunction, exponent: float, type: CrossfadeType):
        super().__init__()
        self.settings["length"] = length if length is not None else 10
        self.settings["function"] = function if function is not None else CrossfadeFunction.power
        self.settings["exponent"] = exponent if exponent is not None else 1.0
        self.settings["type"] = type if type is not None else CrossfadeType.power

        self.__validate__()

    def __validate__(self):
        self.assert_setting_is_number_in_range("length", min_value=0)
        self.assert_setting_is_number_in_range("exponent", min_value=0)

class NoCrossfadeSettings(CrossfadeSettings):

    def __init__(self):
        '''
        This class containes the settings for the NoCrossfade class.
        '''
        super().__init__(length=0, function=CrossfadeFunction.power, exponent=1.0, type=CrossfadeType.power)

class ManualCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = None,
                       function: CrossfadeFunction = None,
                       exponent: float = None,
                       type: CrossfadeType = None):
        '''
        This class containes the settings for the PowerCrossfade class.

            Input:
                length:     length of the crossfade.
                funtion:    function used for the crossfade.
                type:       type of the crossfade.
                exponent:   exponent of the crossfade.
        '''
        super().__init__(length, function, exponent, type)

class LinearCrossfadeSettings(CrossfadeSettings):
    
    def __init__(self, length: int = None, type: CrossfadeType = None):
        '''
        This class containes the settings for the LinearCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__(length=length, function=CrossfadeFunction.power, exponent=1.0, type=type)

class QuadraticCrossfadeSettings(CrossfadeSettings):
    
    def __init__(self, length: int = None, type: CrossfadeType = None):
        '''
        This class containes the settings for the QuadraticCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__(length=length, function=CrossfadeFunction.power, exponent=2.0, type=type)

class CubicCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = None, type: CrossfadeType = None):
        '''
        This class containes the settings for the CubicCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__(length=length, function=CrossfadeFunction.power, exponent=3.0, type=type)

class SinusoidalCrossfadeSettings(CrossfadeSettings):

    def __init__(self, length: int = None, type: CrossfadeType = None):
        '''
        This class containes the settings for the SinusoidalCrossfade class.

            Input:
                length:     length of the crossfade.
        '''
        super().__init__(length=length, function=CrossfadeFunction.sinusoidal, exponent=None, type=type)

class PLCSettings(Settings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None) -> None:
        super().__init__()
        crossfade = [crossfade] if crossfade and not isinstance(crossfade, list) else crossfade
        fade_in = [fade_in] if fade_in and not isinstance(fade_in, list) else fade_in
        self.settings["crossfade_frequencies"] = crossfade_frequencies if crossfade_frequencies is not None else []
        self.__validate_frequencies__()
        self.settings["crossfade"] = crossfade if crossfade is not None else [NoCrossfadeSettings() for _ in range(0, len(self.get("crossfade_frequencies")) + 1)]
        self.settings["fade_in"] = fade_in if fade_in is not None else [NoCrossfadeSettings()]
        self.settings["crossover_order"] = crossover_order if crossover_order is not None else 4

    def __validate_frequencies__(self):
        crossfade_frequencies = self.get("crossfade_frequencies")
        if len(crossfade_frequencies) > 0:
            assert crossfade_frequencies[0] >= 0, "The first crossfade frequency must be greater than or equal to 0."
            assert all(x<y for x, y in zip(crossfade_frequencies, crossfade_frequencies[1:])), "The crossfade frequencies must be in increasing order."
            assert crossfade_frequencies[-1] <= 20000, "The last crossfade frequency must be less than or equal to 20000."

    def __validate__(self):
        self.__validate_frequencies__()
        assert len(self.get("crossfade")) == len(self.get("crossfade_frequencies")) + 1, "The number of crossfade settings must be one more than the number of crossfade frequencies."
        self.assert_setting_is_number_in_range("crossover_order", min_value=1)
    
    def set_crossfade_frequencies(self, crossfade_frequencies: List[int]) -> Settings:
        def change_callback(cloned_settings):
            crossfade_frequencies = cloned_settings.get("crossfade_frequencies")
            crossfade_bands = cloned_settings.get("crossfade")
            new_bands_settings = [crossfade_bands[index] if index < len(crossfade_bands) else NoCrossfadeSettings() for index in range(0, len(crossfade_frequencies) + 1)]
            cloned_settings.settings["crossfade"] = new_bands_settings
        
        return self.__change_setting__("crossfade_frequencies", crossfade_frequencies, change_callback)

class ZerosPLCSettings(PLCSettings):               

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                      fade_in: List[CrossfadeSettings] = None,
                      crossfade_frequencies: List[int] = None,
                      crossover_order: int = None) -> None:
        '''
        This class containes the settings for the ZeroPLC class.
        '''
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)

        self.__validate__()

class ClipStrategy(Enum):
    subtract = "subtract"
    clip = "clip"

class LastPacketPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None,
                       mirror_x: bool = False,
                       mirror_y: bool = False,
                       clip_strategy: ClipStrategy = ClipStrategy.subtract):
        '''
        This class containes the settings for the LastPacketPLC class.
        '''
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)
        self.settings["mirror_x"] = mirror_x
        self.settings["mirror_y"] = mirror_y
        self.settings["clip_strategy"] = clip_strategy

        self.__validate__()


class LowCostPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None,
                       max_frequency: float = 4800,
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
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)
        self.settings["max_frequency"] = max_frequency
        self.settings["f_min"] = f_min
        self.settings["beta"] = beta
        self.settings["n_m"] = n_m
        self.settings["fade_in_length"] = fade_in_length
        self.settings["fade_out_length"] = fade_out_length
        self.settings["extraction_length"] = extraction_length

        self.__validate__()

    def __validate__(self):
        super().__validate__()
        self.assert_setting_is_number_in_range("max_frequency", min_value=0)
        self.assert_setting_is_number_in_range("f_min", min_value=0)
        self.assert_setting_is_number_in_range("beta", min_value=0)
        self.assert_setting_is_number_in_range("n_m", min_value=0)
        self.assert_setting_is_number_in_range("fade_in_length", min_value=0)
        self.assert_setting_is_number_in_range("fade_out_length", min_value=0)
        self.assert_setting_is_number_in_range("extraction_length", min_value=0)
        
        assert self.get("max_frequency") > self.get("f_min"), "The maximum frequency must be greater than the minimum frequency."
        assert not (self.get("fade_in_length") != 0 and self.get("fade_in")[0].__class__ != NoCrossfadeSettings), "The built-in fade_in_length cannot be used in conjunction with the general fade_in mechanism."
        assert not (self.get("fade_out_length") != 0 and self.get("crossfade")[0].__class__ != NoCrossfadeSettings and len(self.get("crossfade")) == 1), "The built-in fade_out_length cannot be used in conjunction with the general crossfade mechanism."

class BurgPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None,
                       context_length: int = 100,
                       order: int = 1):
        '''
        This class containes the settings for the BurgPLC class.

            Input:
                context_length: size of the training set.
                order:          order of the Burg algorithm.
        '''
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)
        self.settings["context_length"] = context_length
        self.settings["order"] = order

        self.__validate__()

    def __validate__(self):
        super().__validate__()
        self.assert_setting_is_number_in_range("context_length", min_value=1)
        self.assert_setting_is_number_in_range("order", min_value=1)

class ExternalPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None):
        '''
        This class containes the settings for the ExternalPLC class.
        '''
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)

class DeepLearningPLCSettings(PLCSettings):

    def __init__(self, crossfade: List[CrossfadeSettings] = None,
                       fade_in: List[CrossfadeSettings] = None,
                       crossfade_frequencies: List[int] = None,
                       crossover_order: int = None,
                       model_path: str = "dl_models/model_bs256_100epochs_0.01_1e-3_1e-7",
                       fs_dl: int = 16000,
                       context_length: int = 8000,
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
        super().__init__(crossfade, fade_in, crossfade_frequencies, crossover_order)
        self.settings["model_path"] = relative_to_root(model_path)
        self.settings["fs_dl"] = fs_dl
        self.settings["context_length"] = context_length
        self.settings["context_length_samples"] = context_length / 1000.0 * self.settings["fs_dl"]
        self.settings["hop_size"] = hop_size
        self.settings["window_length"] = window_length
        self.settings["lower_edge_hertz"] = lower_edge_hertz
        self.settings["upper_edge_hertz"] = upper_edge_hertz
        self.settings["num_mel_bins"] = num_mel_bins

        self.__validate__()

    def __validate__(self):
        super().__validate__()
        self.assert_setting_is_number_in_range("fs_dl", min_value=1)
        self.assert_setting_is_number_in_range("context_length", min_value=1)
        self.assert_setting_is_number_in_range("hop_size", min_value=1)
        self.assert_setting_is_number_in_range("window_length", min_value=1)
        self.assert_setting_is_number_in_range("lower_edge_hertz", min_value=0)
        assert self.get("lower_edge_hertz") < self.get("upper_edge_hertz"), "The lower edge must be less than the upper edge."
        self.assert_setting_is_number_in_range("num_mel_bins", min_value=1)

class AdvancedPLCSettings(PLCSettings):

    def __init__(self, settings: "dict[str, list[PLCSettings]]" = {'linked': [LastPacketPLCSettings(crossfade_frequencies=[3000])]},
                       frequencies: Dict[str, List[int]] = {'linked': []},
                       order: int = 4,
                       stereo_image_processing: StereoImageType = StereoImageType.dual_mono,
                       channel_link: bool = True):
        '''
        This class containes the settings for the AdvancedPLC class.

            Input:
                band_settings:              list of settings for each frequency band.
                frequencies:                list of frequencies used for the crossover (Full band or L/Mid).
                frequencies_b:              list of frequencies used for the crossover (R/Side if unlinked).
                order:                      order of the crossover.
                stereo_image_processing:    type of stereo image processing.
                channel_link:               flag for channel link.
        '''
        Settings.__init__(self)
        self.settings["settings"] = settings
        self.settings["frequencies"] = frequencies
        self.settings["order"] = order
        self.settings["stereo_image_processing"] = stereo_image_processing
        self.settings["channel_link"] = channel_link
        
        self.__validate__()
        
    def __validate__(self):
        keys = set(self.get("settings").keys())
        assert keys == {'linked'} or keys == {'left', 'right'} or keys == {'mid', 'side'}, "The settings must be either linked, left/right or mid/side."
        freq_keys = set(self.get("frequencies").keys())
        assert keys == freq_keys or len(freq_keys) == 0, "If present, the frequencies must be either linked, left/right or mid/side."
        if len(freq_keys) > 0:
            for key in keys:
                assert len(self.get("frequencies")[key]) == len(self.get("settings")[key]) - 1, \
                    f"The number of frequencies for {key} must be one less than the number of settings."

    def set_progress_monitor(self, progress_monitor):
        super().set_progress_monitor(progress_monitor)
        for band_settings in self.settings["settings"].values():
            for settings in band_settings:
                settings.set_progress_monitor(self.progress_monitor)

    def inherit_from(self, parent_settings):
        super().inherit_from(parent_settings)
        for band_settings in self.settings["settings"].values():
            for settings in band_settings:
                settings.inherit_from(parent_settings)
    
    def set_frequencies(self, frequencies: "dict[str, list[int]]") -> Settings:
        def change_callback(cloned_settings):
            channel_settings = cloned_settings.get("settings")
            new_channel_settings = { channel: channel_settings[channel] for channel in frequencies.keys() if channel in channel_settings.keys() }
            for channel, freqs in frequencies.items():
                if channel not in new_channel_settings.keys():
                    new_channel_settings[channel] = [ ZerosPLCSettings() for i in range(0, len(freqs) + 1)]
                else:
                    new_channel_settings[channel] = [ new_channel_settings[channel][i] if i < len(new_channel_settings[channel]) else ZerosPLCSettings() for i in range(0, len(freqs) + 1) ]
            cloned_settings.settings["settings"] = new_channel_settings
            cloned_settings.settings["stereo_image_processing"] = StereoImageType.dual_mono if "left" in new_channel_settings.keys() else StereoImageType.mid_side
            cloned_settings.settings["channel_link"] = new_channel_settings.keys() == {'linked'}
        
        return self.__change_setting__("frequencies", frequencies, change_callback)

    def set_stereo_image_processing(self, stereo_image_processing: StereoImageType) -> Settings:
        def change_callback(cloned_settings):
            frequencies = cloned_settings.get("frequencies")
            channel_settings = cloned_settings.get("settings")
            keys = set(channel_settings.keys())
            if keys == {'linked'}:
                return cloned_settings  
            if keys == {'left', 'right'}:
                frequencies = { "mid" if k == "left" else "side": v for k, v in frequencies.items()}
                channel_settings = { "mid" if k == "left" else "side": v for k, v in channel_settings.items()}
            elif keys == {'mid', 'side'}:
                frequencies = { "left" if k == "mid" else "right": v for k, v in frequencies.items()}
                channel_settings = { "left" if k == "mid" else "right": v for k, v in channel_settings.items()}
            cloned_settings.settings["frequencies"] = frequencies
            cloned_settings.settings["settings"] = channel_settings
        
        return self.__change_setting__("stereo_image_processing", stereo_image_processing, change_callback)
    
    def set_channel_link(self, channel_link) -> Settings:
        def change_callback(cloned_settings):
            frequencies = cloned_settings.get("frequencies")
            channel_settings = cloned_settings.get("settings")
            if channel_link:
                frequencies = {"linked": next(iter(frequencies.values()))}
                channel_settings = {"linked": next(iter(channel_settings.values()))}
            else:
                stereo_image_processing = cloned_settings.get("stereo_image_processing")
                frequencies = {
                    "left" if stereo_image_processing == StereoImageType.dual_mono else "mid" : next(iter(frequencies.values())),
                    "right" if stereo_image_processing == StereoImageType.dual_mono else "side" : next(iter(frequencies.values())),
                }
                channel_settings = {
                    "left" if stereo_image_processing == StereoImageType.dual_mono else "mid" : next(iter(channel_settings.values())),
                    "right" if stereo_image_processing == StereoImageType.dual_mono else "side" : next(iter(channel_settings.values())),
                }
            cloned_settings.settings["frequencies"] = frequencies
            cloned_settings.settings["settings"] = channel_settings
        
        return self.__change_setting__("channel_link", channel_link, change_callback)


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
    basic = "basic"
    advanced = "advanced"

class PEAQCalculatorSettings(Settings):

    def __init__(self, peaq_mode: PEAQMode = PEAQMode.basic):
        '''
        This class containes the settings for the PEAQCalculator class.

            Input:
                peaq_mode:      mode of the PEAQ algorithm.
        '''
        super().__init__()
        self.settings["peaq_mode"] = peaq_mode

class PerceptualCalculatorSettings(Settings):

    def __init__(self, intorno_length: int = 300,
                       min_frequency: float = 32.7,
                       max_frequency: float = 20000,
                       bins_per_octave: int = 12,
                       minimum_window: int = 128,
                       masking: bool = True) -> None:
        super().__init__()
        self.settings["intorno_length"] = intorno_length
        self.settings["min_frequency"] = min_frequency
        self.settings["max_frequency"] = max_frequency
        self.settings["bins_per_octave"] = bins_per_octave
        self.settings["minimum_window"] = minimum_window
        self.settings["masking"] = masking

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

