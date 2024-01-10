from scipy.signal import iirfilter, sosfilt
from .utils import force_2d

class LinkwitzRileyFilter:
    def __init__(self, order, cutoff_frequency, sampling_rate, type='low'):
        self.order = order
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        self.type = type
        self.sos = self._design_filter()

    def _design_filter(self):
        nyquist_frequency = 0.5 * self.sampling_rate
        normalized_cutoff_frequency = self.cutoff_frequency / nyquist_frequency
        sos = iirfilter(N=self.order, Wn=normalized_cutoff_frequency, btype=self.type, ftype='butter', output='sos')
        return sos

    def filter(self, data):
        return force_2d(sosfilt(self.sos, data))
    
class LinkwitzRileyCrossover:
    def __init__(self, order, cutoff_frequency, sampling_rate):
        self.order = order
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        self.hp_filter = LinkwitzRileyFilter(self.order, self.cutoff_frequency, self.sampling_rate, type='high')
        self.lp_filter = LinkwitzRileyFilter(self.order, self.cutoff_frequency, self.sampling_rate, type='low')

    def split(self, data):
        hp_data = self.hp_filter.filter(data)
        lp_data = self.lp_filter.filter(data)
        return hp_data, lp_data