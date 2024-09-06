from .settings import Settings
from .utils import relative_to_root

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from essentia.standard import NSGConstantQ
import librosa
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from brian2hears import LogGammachirp, RestructureFilterbank, AsymmetricCompensation, asymmetric_compensation_coeffs, ControlFilterbank, erbspace, Sound
from brian2 import Hz, kHz, ms, log10, mean, diff, asarray, minimum, maximum, arange, exp, log

import cProfile
import pstats
import io

def S1dataset_generateTFmaskfunc(center_f, f_axis, ERBspac=1, timespac=0.001, varargin=[8]):
    
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
    
    data = np.load(relative_to_root('masking_data/S1dataset_rawdata.npz'))
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

class DCGC(object):

    def __init__(self) -> None:
        pass

    def __call__(self, original_sound, glitch_sound, nbr_cf=100, min_freq=20*Hz, max_freq=20000*Hz) -> dict:

        self.samplerate = original_sound.samplerate
        self.nbr_cf = nbr_cf # number of centre frequencies
        # center frequencies with a spacing following an ERB scale
        self.cf = erbspace(min_freq, max_freq, self.nbr_cf)

        self.c1 = -2.96 #glide slope of the first filterbank
        self.b1 = 1.81  #factor determining the time constant of the first filterbank
        self.c2 = 2.2   #glide slope of the second filterbank
        self.b2 = 2.17  #factor determining the time constant of the second filterbank

        self.order_ERB = 4
        self.ERBrate = 21.4*log10(4.37*(self.cf/kHz)+1)
        self.ERBwidth = 24.7*(4.37*(self.cf/kHz) + 1)
        self.ERBspace = mean(diff(self.ERBrate))

        # the filter coefficients are updated every update_interval (here in samples)
        self.update_interval = 1

        #bank of passive gammachirp filters. As the control path uses the same passive
        #filterbank than the signal path (but shifted in frequency)
        #this filterbank is used by both pathway.

        self.fp1 = asarray(self.cf) + self.c1*self.ERBwidth*self.b1/self.order_ERB #centre frequency of the signal path

        corrupted_sound = original_sound + glitch_sound
        self.pGc_control = LogGammachirp(corrupted_sound, self.cf, b=self.b1, c=self.c1)
        self.control_path()
        self.pGc_signal = LogGammachirp(original_sound, self.cf, b=self.b1, c=self.c1)
        self.signal_path()
        self.original_signal = self.signal

        self.pGc_control = LogGammachirp(corrupted_sound, self.cf, b=self.b1, c=self.c1)
        self.control_path()
        self.pGc_signal = LogGammachirp(glitch_sound, self.cf, b=self.b1, c=self.c1)
        self.signal_path()
        self.glitch_signal = self.signal
        return {'original': self.original_signal, 'reconstructed': self.glitch_signal}


    def control_path(self):

        #### Control Path ####

        #the first filterbank in the control path consists of gammachirp filters
        #value of the shift in ERB frequencies of the control path with respect to the signal path
        self.lct_ERB = 1.5
        self.n_ch_shift = round(self.lct_ERB/self.ERBspace) #value of the shift in channels
        #index of the channel of the control path taken from pGc
        self.indch1_control = minimum(maximum(1, arange(1, self.nbr_cf+1)+self.n_ch_shift), self.nbr_cf).astype(int)-1
        self.fp1_control = self.fp1[self.indch1_control]
        #the control path bank pass filter uses the channels of pGc indexed by indch1_control
        self.pGc_control = RestructureFilterbank(self.pGc_control, indexmapping=self.indch1_control)

        #the second filterbank in the control path consists of fixed asymmetric compensation filters
        self.frat_control = 1.08
        self.fr2_control = self.frat_control*self.fp1_control
        self.asym_comp_control = AsymmetricCompensation(self.pGc_control, self.fr2_control, b=self.b2, c=self.c2)

        #definition of the pole of the asymmetric comensation filters
        self.p0 = 2
        self.p1 = 1.7818*(1-0.0791*self.b2)*(1-0.1655*abs(self.c2))
        self.p2 = 0.5689*(1-0.1620*self.b2)*(1-0.0857*abs(self.c2))
        self.p3 = 0.2523*(1-0.0244*self.b2)*(1+0.0574*abs(self.c2))
        self.p4 = 1.0724

        #definition of the parameters used in the control path output levels computation
        #(see IEEE paper for details)
        self.decay_tcst = .5*ms
        self.order = 1.
        self.lev_weight = .5
        self.level_ref = 50.
        self.level_pwr1 = 1.5
        self.level_pwr2 = .5
        self.RMStoSPL = 30.
        self.frat0 = .2330
        self.frat1 = .005
        self.exp_deca_val = exp(-1/(self.decay_tcst*self.samplerate)*log(2))
        self.level_min = 10**(-self.RMStoSPL/20)

    #### Signal Path ####
    #the signal path consists of the passive gammachirp filterbank pGc previously
    #defined followed by a asymmetric compensation filterbank
    def signal_path(self):

        #definition of the controller class. What is does it take the outputs of the
        #first and second fitlerbanks of the control filter as input, compute an overall
        #intensity level for each frequency channel. It then uses those level to update
        #the filter coefficient of its target, the asymmetric compensation filterbank of
        #the signal path.
        class CompensensationFilterUpdater(object):
            def __init__(self, target, dcgc):
                self.target = target
                self.level1_prev = -100
                self.level2_prev = -100
                self.dcgc = dcgc

            def __call__(self, *input):
                value1 = input[0][-1,:]
                value2 = input[1][-1,:]
                #the current level value is chosen as the max between the current
                #output and the previous one decreased by a decay
                level1 = maximum(maximum(value1, 0), self.level1_prev*self.dcgc.exp_deca_val)
                level2 = maximum(maximum(value2, 0), self.level2_prev*self.dcgc.exp_deca_val)

                self.level1_prev = level1 #the value is stored for the next iteration
                self.level2_prev = level2
                #the overall intensity is computed between the two filterbank outputs
                level_total = self.dcgc.lev_weight*self.dcgc.level_ref*(level1/self.dcgc.level_ref)**self.dcgc.level_pwr1+\
                        (1-self.dcgc.lev_weight)*self.dcgc.level_ref*(level2/self.dcgc.level_ref)**self.dcgc.level_pwr2
                #then it is converted in dB
                level_dB = 20*log10(maximum(level_total, self.dcgc.level_min))+self.dcgc.RMStoSPL
                #the frequency factor is calculated
                frat = self.dcgc.frat0 + self.dcgc.frat1*level_dB
                #the centre frequency of the asymmetric compensation filters are updated
                fr2 = self.dcgc.fp1*frat
                coeffs = asymmetric_compensation_coeffs(self.dcgc.samplerate, fr2,
                                self.target.filt_b, self.target.filt_a, self.dcgc.b2, self.dcgc.c2,
                                self.dcgc.p0, self.dcgc.p1, self.dcgc.p2, self.dcgc.p3, self.dcgc.p4)
                self.target.filt_b, self.target.filt_a = coeffs

        self.fr1 = self.fp1*self.frat0
        varyingfilter_signal_path = AsymmetricCompensation(self.pGc_signal, self.fr1, b=self.b2, c=self.c2)
        updater = CompensensationFilterUpdater(varyingfilter_signal_path, self)
        #the controler which takes the two filterbanks of the control path as inputs
        #and the varying filter of the signal path as target is instantiated
        control = ControlFilterbank(varyingfilter_signal_path,
                                    [self.pGc_control, self.asym_comp_control],
                                    varyingfilter_signal_path, updater, self.update_interval)

        #run the simulation
        #Remember that the controler are at the end of the chain and the output of the
        #whole path comes from them
        self.signal = control.process()

class PerceptualMetric(object):

    def __init__(self, transform_type: str,
                       min_frequency: float,
                       max_frequency: float,
                       bins_per_octave: int,
                       n_bins: int,
                       minimum_window: int,
                       input_size: int,
                       fs: int,
                       intorno_length: int,
                       linear_mag: bool,
                       masking: bool,
                       masking_offset: int,
                       db_weighting: str,
                       metric: str) -> None:
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins
        self.minimum_window = minimum_window
        self.input_size = input_size
        self.fs = fs

        if transform_type == 'cqt':
            cqt = NSGConstantQ(minFrequency=min_frequency,
                               maxFrequency=max_frequency,
                               binsPerOctave=bins_per_octave,
                               minimumWindow=minimum_window,
                               inputSize=input_size)
            self.transform = lambda original, reconstructed : {'original': cqt(original)[0], 'reconstructed': cqt(reconstructed)[0]}
            self.frequency_axis = librosa.cqt_frequencies(input_size, fmin=min_frequency, bins_per_octave=bins_per_octave)
        elif transform_type == 'dcgc':
            dcgc = DCGC()
            self.transform = lambda original, reconstructed: dcgc(Sound(original, samplerate=self.fs*Hz), Sound(reconstructed, samplerate=self.fs*Hz), nbr_cf=self.n_bins, min_freq=self.min_frequency*Hz, max_freq=self.max_frequency*Hz)
            self.frequency_axis = erbspace(self.min_frequency*Hz, self.max_frequency*Hz, self.bins_per_octave)

        self.intorno_length = intorno_length
        self.linear_mag = linear_mag
        self.masking = masking
        self.masking_offset = masking_offset
        self.db_weighting = db_weighting
        self.metric = metric
        

    def spectrogram(self, original, reconstructed):
        return self.transform(original, reconstructed)

    def __call__(self, spectrogram):
        if self.linear_mag:
            spectrogram_original_mag = np.abs(spectrogram['original'])
            spectrogram_reconstructed_mag = np.abs(spectrogram['reconstructed'])

            spectrogram_reconstructed_residual = spectrogram_reconstructed_mag - spectrogram_original_mag
            spectrogram_reconstructed_residual = np.where(spectrogram_reconstructed_residual < 0, 0, spectrogram_reconstructed_residual)
            
            perc_metric = np.sum(spectrogram_reconstructed_residual) / 10

        else:
            spectrogram_difference = spectrogram['reconstructed'] - spectrogram['original']
            spectrogram_difference_mag = np.abs(spectrogram_difference)
            ref_value = max(np.max(np.abs(spectrogram['original'])), np.max(np.abs(spectrogram['reconstructed'])))
            spectrogram_original_db = librosa.amplitude_to_db(np.abs(spectrogram['original']), ref=ref_value)
            spectrogram_difference_db = librosa.amplitude_to_db(spectrogram_difference_mag, ref=ref_value)

            freq_axis = librosa.cqt_frequencies(spectrogram_original_db.shape[0], fmin=self.min_frequency, bins_per_octave=self.bins_per_octave)

            if self.masking:
                # Profile the function
                # pr = cProfile.Profile()
                # pr.enable()

                mask = apply_masking_to_cqt(spectrogram_original_db, freq_axis, self.fs, self.intorno_length) + self.masking_offset

                # pr.disable()
                
                # Collect profiling results
                # s = io.StringIO()
                # sortby = 'cumulative'
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()

                # Output profiling results
                # print(s.getvalue())

                spectrogram_difference_masked = np.where(spectrogram_difference_db < mask, 0, spectrogram_difference)
                spectrogram_difference_masked_mag = np.abs(spectrogram_difference_masked)
                spectrogram_difference_masked_db = librosa.amplitude_to_db(spectrogram_difference_masked_mag, ref=ref_value)
            else:
                spectrogram_difference_masked = spectrogram['reconstructed']
                spectrogram_difference_masked_db = spectrogram_difference_db

            if self.db_weighting == 'A':
                a_weigthing = librosa.A_weighting(freq_axis)[:, np.newaxis]
                spectrogram_difference_masked_db = spectrogram_difference_masked_db + a_weigthing
                spectrogram_original_db = spectrogram_original_db + a_weigthing
            if self.db_weighting == 'C':
                c_weigthing = librosa.C_weighting(freq_axis)[:, np.newaxis]
                spectrogram_difference_masked_db = spectrogram_difference_masked_db + c_weigthing
                spectrogram_original_db = spectrogram_original_db + c_weigthing

            spectrogram_original_loss_db = np.where(np.abs(spectrogram_difference_masked) > 0, spectrogram_original_db, 0)
            spectrogram_reconstructed_masked_residual = spectrogram_difference_masked_db - spectrogram_original_loss_db

            if self.metric == 'weighted_sum':
                avg = (spectrogram_difference_masked_db + spectrogram_original_loss_db)/2
                complement = 100 - avg
                spectrogram_reconstructed_masked_residual = spectrogram_reconstructed_masked_residual * complement
                
            spectrogram_reconstructed_masked_residual = np.where(spectrogram_reconstructed_masked_residual < 0, 0, spectrogram_reconstructed_masked_residual)
            perc_metric = np.sum(spectrogram_reconstructed_masked_residual) / 10000

        print(f"{perc_metric}")

        # time_axis = np.linspace(0, spectrogram_original_db.shape[1] / self.fs, num=spectrogram_original_db.shape[1])

        # plot_idx = 18699
        # if spectrogram['idx'] == plot_idx + 1:
        #     # plt.figure(figsize=(12, 4))
        #     # plt.imshow(cqt_original_db, aspect='auto', origin='lower', cmap='jet')
        #     # plt.colorbar()
        #     # plt.title('Original CQT')
        #     # plt.savefig('original_cqt.png')
        #     note_axis = [librosa.hz_to_note(f) for f in freq_axis]
        #     fig = go.Figure()
        #     fig.add_trace(go.Surface(x=time_axis, y=note_axis, z=spectrogram_difference_masked_db, colorscale='Inferno'))
        #     fig.add_trace(go.Surface(x=time_axis, y=note_axis, z=spectrogram_original_db, colorscale='Viridis'))
        #     fig.update_layout(scene = dict(
        #                 xaxis_title='Time',
        #                 yaxis_title='CQT bins',
        #                 zaxis_title='Amplitude (dB)'),
        #               title="Interactive 3D CQT Plot")
        #     fig.show()

        
        return perc_metric