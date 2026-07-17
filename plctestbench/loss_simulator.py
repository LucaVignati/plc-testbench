import numpy as np
import numpy.random as npr

from plctestbench.worker import Worker

from .settings import (
    BinomialPLSSettings,
    GilbertElliotPLSSettings,
    MetronomePLSSettings,
    CustomMaskPLSSettings,
    Settings,
)


class PacketLossSimulator(Worker):
    """
    Base class for all the loss models.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Variables:
            seed:   value to be used as a seed for the random number
                    generator.
        """
        super().__init__(settings)
        self.packet_size = settings.get("packet_size")

    def run(self, num_samples, id) -> np.ndarray:
        """
        This function computes and returns an array of indexes representing
        the position of lost samples in the original audio track.
        """
        lost_samples_idx = []
        for idx in self.progress_monitor(range(num_samples), desc=f"{str(self)}|{id}"):
            if (idx % self.packet_size) == 0:
                lost_packet = self.tick()
            if lost_packet:
                lost_samples_idx.append(idx)

        return np.array(lost_samples_idx)

    def __str__(self) -> str:
        return self.__class__.__name__ + "_s" + str(self.settings.get("seed"))

    def tick(self) -> bool:
        """
        Placeholder function to be implemented by the derived classes.
        """
        raise NotImplementedError


class BinomialPLS(PacketLossSimulator):
    """
    This class implements a binomial distribution to be used as a loss model
    in packet/sample loss simulators.
    """

    def __init__(self, settings: BinomialPLSSettings) -> None:
        """
        Variables:
            per:    the Packet Error Ratio is the ratio between the lost packets
                    and the total number of packets.
        """
        super().__init__(settings)
        self.per = settings.get("per")
        npr.seed(self.settings.get("seed"))

    def tick(self) -> bool:
        """
        This function performs a Bernoulli trial and returns the result.
        Output:
            True if the packet has been lost
        """
        b_trial_result = npr.random() <= self.per
        return b_trial_result


class MetronomePLS(PacketLossSimulator):
    """
    This class implements a deterministic and periodic
    metronome packet loss model.


    The model generates bursts of consecutive packet losses at regular
    intervals. Each period defines a cycle in which a fixed number of
    packets are dropped, followed by a sequence of correctly received
    packets.

    The simulator operates on a per-packet basis (via `tick()`), maintaining
    an internal counter that advances at each packet boundary.


    For each cycle:
    - The cycle length is defined by `period`
    - The first `duration` packets in the cycle are marked as lost
    - The remaining packets in the cycle are received correctly


    An optional `offset` allows delaying the start of the first loss burst,
    effectively shifting the pattern in time.


    Example (period=10, duration=3, offset=0):
        Packet indices:   0 1 2 3 4 5 6 7 8 9 | 10 11 12 ...
        Loss pattern:     L L L R R R R R R R | L  L  L ...


    Example (period=10, duration=3, offset=4):
        Packet indices:   0 1 2 3 4 5 6 7 8 9 | 10 11 ...
        Loss pattern:     R R R R L L L R R R | R  L ...


    """

    def __init__(self, settings: MetronomePLSSettings) -> None:
        """
        period : int
            Length of the full cycle in packets (loss + no-loss).
            Must be >= 1.


        duration : int
            Number of consecutive packets lost at the start of each cycle.
            Must satisfy 0 <= duration <= period.


        offset : int
            Initial shift (in packets) applied before the periodic pattern starts.
            A positive offset delays the first loss burst.


        counter : int
            Internal state tracking the current position within the cycle.
            Initialized as `-offset` to account for the initial delay.
        """
        super().__init__(settings)
        self.period = settings.get("period")
        self.duration = settings.get("duration")
        self.offset = settings.get("offset")
        self.counter = -self.offset

    def tick(self) -> bool:
        """
        Advances the internal counter and determines whether the current
        packet is lost.

        Returns
        -------
        bool
            True if the current packet is lost, False otherwise.
        """
        self.counter += 1
        if self.counter == self.period:
            self.counter = 0
        if self.counter < self.duration and self.counter >= 0:
            return True
        return False


class GilbertElliotPLS(PacketLossSimulator):
    """
    This class implements the Gilbert-Elliott packet loss model.


    Adapted from:

    https://github.com/mkalewski/sim2net/blob/master/sim2net/packet_loss/gilbert_elliott.py

    (MIT licensed)
    """

    def __init__(self, settings: GilbertElliotPLSSettings) -> None:
        """
        Variables:
            p: probability to transition from GOOD to BAD
            r: probability to transition from BAD to GOOD
            h: probability of a good packet in BAD state
            k: probability of a good packet in a GOOD state
        """
        super().__init__(settings)
        npr.seed(self.settings.get("seed"))
        p = settings.get("p")
        r = settings.get("r")
        h = settings.get("h")
        k = settings.get("k")

        b = 1.0 - h
        g = 1.0 - k
        # ( current state: 'G' or 'B',
        #   transition probability,
        #   current packet error rate )
        self.state_g = ("G", p, g)
        self.state_b = ("B", r, b)
        self.current_state = self.state_g

    def tick(self) -> bool:
        """
        Returns information about whether a transmitted packet has been lost or
        can be successfully received by destination node(s) according to the
        Gilbert-Elliott packet loss model.
        Output:
            True if the packet has been lost
        """
        transition = npr.random()
        if transition <= self.current_state[1]:
            if self.current_state[0] == "G":
                self.current_state = self.state_b
            else:
                self.current_state = self.state_g
        loss = npr.random()
        if loss <= self.current_state[2]:
            return True
        return False


class CustomMaskPLS(PacketLossSimulator):
    """
    Packet loss simulator that uses a predefined binary mask to determine
    which packets are considered lost.

    Each packet corresponds to one character in the mask string:
    - `'1'` indicates a lost packet (by default).
    - `'0'` indicates a successfully received packet.
    This behavior can be inverted by setting `invert=True`.

    The simulator increments an internal counter at every tick. When the
    counter exceeds the length of the mask, no further packets are dropped
    (i.e., all subsequent packets are considered received).

    It is recommended that the total number of packets ('#samples / packet_size')
    match the length of the mask. If they differ, the simulator stops applying
    the mask once the end is reached.

    Attributes
    ----------
    mask : str
        A string of '0' and '1' characters defining the drop pattern.
    invert : bool
        If True, inverts the meaning of mask values ('1' = not lost, '0' = lost).
    counter : int
        Internal counter tracking the current mask position.

    Methods
    -------
    tick() -> bool
        Returns True if the current packet should be considered lost,
        False otherwise. Automatically advances the internal counter.
    """

    def __init__(self, settings: CustomMaskPLSSettings):
        super().__init__(settings)
        self.mask: str = settings.get("mask")
        self.invert: bool = settings.get("invert")
        self.counter = 0

    def tick(self):
        try:
            lost = bool(int(self.mask[self.counter]))
            lost = not lost if self.invert else lost
        except IndexError:
            lost = False
        except ValueError:
            lost = False
        self.counter += 1
        return lost
