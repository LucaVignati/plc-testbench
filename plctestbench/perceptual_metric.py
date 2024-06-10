from .settings import Settings

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from essentia.standard import NSGConstantQ
import librosa

def extract_intorni(audio_file, lost_samples_idxs, intorno_size, fs, packet_size):
    def find_boundary_indexes(lost_samples_idxs):
        boundary_indexes = []
        start_idx = None
        end_idx = None
        for idx in lost_samples_idxs:
            if start_idx is None:
                start_idx = idx
                end_idx = idx
            elif idx == end_idx + 1:
                end_idx = idx
            else:
                boundary_indexes.append((start_idx, end_idx))
                start_idx = None
            if idx == lost_samples_idxs[-1]:
                boundary_indexes.append((start_idx, end_idx))
        return boundary_indexes

    intorno_samples = float(intorno_size*fs)/1000
    audio_data = audio_file.get_data()
    boundary_indexes = find_boundary_indexes(lost_samples_idxs)
    intorni = []
    packet_idxs = []
    for start_idx, end_idx in boundary_indexes:
        center_idx = (start_idx + end_idx) // 2
        start_sample = int(center_idx - intorno_samples // 2)
        end_sample = int(center_idx + intorno_samples // 2)
        intorno = audio_data[start_sample:end_sample]
        intorni.append(intorno)
        packet_idxs.append(center_idx//packet_size)
    return packet_idxs, intorni

def S1dataset_generateTFmaskfunc(center_f, f_axis, plot=False, ERBspac=1, timespac=0.001, varargin=[8]):
    
    def freq_to_erb(freq):
        return 21.4 * np.log10(1 + freq / 229)

    def erb_to_hz(erb):
        """
        Convert from equivalent rectangular bandwidth (ERB) scale to hertz.
        
        Parameters:
        erb (float): Input frequency on the ERB scale.
        
        Returns:
        float: Output frequency in Hz.
        """
        return (np.exp(erb / 9.26449) - 1) * 24.7 * 9.26449
    
    data = np.load('masking_data/S1dataset_rawdata.npz')
    dT = data['dT'].flatten()
    dF = data['dF'].flatten()
    AM = data['data']

    if varargin is None:
        ERBmin = np.min(dF)
        ERBmax = np.max(dF)
    elif len(varargin) == 1:
        lim = varargin[0]
        if np.isscalar(lim):
            ERBmin = -lim
            ERBmax = lim
    else:
        if len(varargin) == 2:
            ERBmin = varargin[0]
            ERBmax = varargin[1]
        else:
            raise ValueError('More than two limits were provided!')

    dF_hz = erb_to_hz(freq_to_erb(center_f)+dF)

    dFi = erb_to_hz(freq_to_erb(center_f)+np.arange(ERBmin, ERBmax + ERBspac, ERBspac))
    dTi = np.arange(np.min(dT), np.max(dT) + timespac, timespac)
    
    freqs = f_axis[(f_axis >= np.min(dFi)) & (f_axis <= np.max(dFi))]

    r = RGI((dT, dF_hz), AM.T, method='linear', bounds_error=False, fill_value=0)
    dTi_grid, dFi_grid = np.meshgrid(dTi, freqs, indexing='ij')
    AMi = r((dTi_grid, dFi_grid)) - 60

    return freqs, dTi, AMi

def apply_masking_to_cqt(cqt_mag, freq_axis, fs, duration, masked_intorno_duration=(150, 150)):
    
    assert sum(masked_intorno_duration) <= duration
    
    num_bins, num_frames = cqt_mag.shape
    hop_duration = duration/num_frames
    hop_length = int(hop_duration/1000*fs)
    masked_intorno_hops = np.round(np.array(masked_intorno_duration)/hop_duration).astype(int)
    cqt_intorno_mag = cqt_mag[:, int(num_frames/2-masked_intorno_hops[0]):int(num_frames/2+masked_intorno_hops[1])]
    _, num_frames_intorno = cqt_intorno_mag.shape
    masked_cqt = np.full(cqt_intorno_mag.shape, -100)

    for bin_idx in range(num_bins):
        center_f = freq_axis[bin_idx]
        dFi, dTi, AMi = S1dataset_generateTFmaskfunc(center_f, freq_axis, timespac=1/fs*hop_length)

        kernels = AMi.T[np.newaxis, :, :] * np.ones((num_frames_intorno, len(dFi), len(dTi)))
        masker_levels = cqt_intorno_mag[bin_idx, :, np.newaxis, np.newaxis] * np.ones((num_frames_intorno, len(dFi), len(dTi)))
        thresholds = masker_levels + kernels
        thresholds = np.pad(thresholds, ((0, 0), (0, 0), (0, num_frames_intorno)), 'constant', constant_values=-100)
        for frame_idx in range(num_frames_intorno):
            thresholds[frame_idx, :, :] = np.roll(thresholds[frame_idx, :, :], frame_idx, axis=1)
        masked_levels = np.max(thresholds, axis=0)

        # Find the frequency bins and time indices within the masking range
        freq_indices = np.where((freq_axis >= np.min(dFi)) & (freq_axis <= np.max(dFi)))[0]

        masked_cqt[freq_indices, :] = np.maximum(masked_cqt[freq_indices, :], masked_levels[:, :num_frames_intorno])

    mask = np.full(cqt_mag.shape, -100)
    mask[:, int(num_frames/2-masked_intorno_hops[0]):int(num_frames/2+masked_intorno_hops[1])] = masked_cqt
    return mask

class PerceptualMetric(object):

    def __init__(self, min_frequency,
                       max_frequency,
                       bins_per_octave,
                       minimum_window,
                       input_size,
                       fs,
                       intorno_length) -> None:
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.bins_per_octave = bins_per_octave
        self.minimum_window = minimum_window
        self.input_size = input_size
        self.fs = fs
        self.intorno_length = intorno_length
        self.CQT = NSGConstantQ(minFrequency=self.min_frequency,
                                maxFrequency=self.max_frequency,
                                binsPerOctave=self.bins_per_octave,
                                minimumWindow=self.minimum_window,
                                inputSize=self.input_size)

    def cqt(self, audio_data):
        return self.CQT(audio_data)

    def __call__(self, cqt):
        ref_value = max(np.max(np.abs(cqt['original'])), np.max(np.abs(cqt['difference'])))
        cqt_original_db = librosa.amplitude_to_db(np.abs(cqt['original']), ref=ref_value)
        cqt_difference_db = librosa.amplitude_to_db(np.abs(cqt['difference']), ref=ref_value)

        freq_axis = librosa.cqt_frequencies(cqt_original_db.shape[0], fmin=self.min_frequency, bins_per_octave=self.bins_per_octave)

        mask = apply_masking_to_cqt(cqt_original_db, freq_axis, self.fs, self.intorno_length)

        cqt_difference_masked = np.where(cqt_difference_db < mask, 0, cqt['difference'])
        cqt_difference_masked_mag = np.abs(cqt_difference_masked)
        cqt_difference_masked_db = librosa.amplitude_to_db(cqt_difference_masked_mag, ref=ref_value)

        cqt_original_loss_db = np.where(cqt_difference_masked > 0, cqt_original_db, 0)
        cqt_difference_masked_db_residual = cqt_difference_masked_db - cqt_original_loss_db
        cqt_difference_masked_db_residual = np.where(cqt_difference_masked_db_residual < 0, 0, cqt_difference_masked_db_residual)
        perc_metric = np.sum(cqt_difference_masked_db_residual) / 10000
        return perc_metric