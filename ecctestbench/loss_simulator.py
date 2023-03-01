import numpy as np
import numpy.random as npr
from tqdm.notebook import tqdm
from ecctestbench.worker import Worker
from .settings import Settings

class LossSimulator(Worker):
    '''
    Base class for all the loss models.
    '''
    def __init__(self, settings: Settings) -> None:
        '''
        Variables:
            seed:   value to be used as a seed for the random number
                    generator.
        '''
        self.settings = settings
        self.packet_size = settings.packet_size

    def run(self, num_samples) ->np.ndarray:
        '''
        This function computes and returns an array of indexes representing
        the position of lost samples in the original audio track.
        '''
        lost_samples_idx = []
        for idx in tqdm(range(num_samples), desc=self.__str__()):
            if (idx % self.packet_size) == 0:
                lost_packet = self.tick()
            if lost_packet:
                lost_samples_idx.append(idx)

        return np.array(lost_samples_idx)

    def tick(self) -> bool:
        '''
        Placeholder function to be implemented by the derived classes.
        '''
        pass

class BinomialLossSimulator(LossSimulator):
    '''
    This class implements a binomial distribution to be used as a loss model
    in packet/sample loss simulators.
    '''

    def __init__(self, settings: Settings) -> None:
        '''
        Variables:
            per:    the Packet Error Ratio is the ratio between the lost packets
                    and the total number of packets.
        '''
        super().__init__(settings)
        self.per = settings.per
        npr.seed(self.settings.seed)


    def tick(self) -> bool:
        '''
        This function performs a Bernoulli trial and returns the result.
        Output:
            True if the packet has been lost
        '''
        b_trial_result = npr.random() <= self.per
        return b_trial_result

    def __str__(self) -> str:
        return __class__.__name__ + '_s' + str(self.settings.seed)

class GilbertElliotLossSimulator(LossSimulator):
    '''
    This class implements the Gilbert-Elliott packet loss model.


    Adapted from:

    https://github.com/mkalewski/sim2net/blob/master/sim2net/packet_loss/gilbert_elliott.py

    (MIT licensed)
    '''
    def __init__(self, settings: Settings) -> None:
        '''
        Variables:
            p: probability to transition from GOOD to BAD
            r: probability to transition from BAD to GOOD
            h: probability of a good packet in BAD state
            k: probability of a good packet in a GOOD state
        '''
        super().__init__(settings)
        npr.seed(self.settings.seed)
        p = settings.p
        r = settings.r
        h = settings.h
        k = settings.k

        b = 1.0 - h
        g = 1.0 - k
        # ( current state: 'G' or 'B',
        #   transition probability,
        #   current packet error rate )
        self.state_g = ('G', p, g)
        self.state_b = ('B', r, b)
        self.current_state = self.state_g

    def tick(self) -> bool:
        '''
        Returns information about whether a transmitted packet has been lost or
        can be successfully received by destination node(s) according to the
        Gilbert-Elliott packet loss model.
        Output:
            True if the packet has been lost
        ''' 
        transition = npr.random()
        if transition <= self.current_state[1]:
            if self.current_state[0] == 'G':
                self.current_state = self.state_b
            else:
                self.current_state = self.state_g
        loss = npr.random()
        if loss <= self.current_state[2]:
            return True
        return False

    def __str__(self) -> str:
        return __class__.__name__ + '_s' + str(self.settings.seed)