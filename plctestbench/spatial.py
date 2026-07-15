from enum import Enum

import numpy as np


class CodecMode(Enum):
    ENCODE = 0
    DECODE = 1


class MidSideCodec(object):
    """Convert stereo audio between left/right and mid/side representations.

    The codec expects a 2D NumPy array with shape ``(num_samples, 2)``.

    - In :attr:`CodecMode.ENCODE` mode, the input is interpreted as
      ``[left, right]`` and converted to ``[mid, side]``.
    - In :attr:`CodecMode.DECODE` mode, the input is interpreted as
      ``[mid, side]`` and converted back to ``[left, right]``.

    The transform used by this implementation is:

    - ``mid = (left + right) / 2``
    - ``side = (left - right) / 2``
    - ``left = (mid + side) * 2``
    - ``right = (mid - side) * 2``

    This decode implementation applies a factor of ``2`` after recombination,
    matching the scaling used in the provided code.

    Example:
        >>> codec = MidSideCodec()
        >>> stereo = np.array([[1.0, 0.5], [0.2, -0.2]])
        >>> mid_side = codec(stereo, CodecMode.ENCODE)
        >>> restored = codec(mid_side, CodecMode.DECODE)
    """

    def __call__(
        self, audio: np.ndarray, type: CodecMode = CodecMode.ENCODE
    ) -> np.ndarray:
        """Apply mid/side encoding or decoding to a stereo audio buffer.

        Args:
            audio: Input audio array of shape ``(num_samples, 2)``.
                For encoding, columns represent left and right channels.
                For decoding, columns represent mid and side channels.
            type: Codec operation mode. Defaults to :attr:`CodecMode.ENCODE`.

        Returns:
            A NumPy array of shape ``(num_samples, 2)`` containing the
            transformed audio channels.

        Raises:
            ValueError: If ``type`` is not a supported :class:`CodecMode`.
        """

        if type == CodecMode.ENCODE:
            return MidSideCodec.encode(audio)
        elif type == CodecMode.DECODE:
            return MidSideCodec.decode(audio)
        else:
            raise ValueError("MidSideCodec: Unsupported CodecMode '%s'", type)

    @staticmethod
    def encode(left_right: np.ndarray) -> np.ndarray:
        """Encode left/right stereo audio into mid/side format.

        Args:
            left_right: Stereo audio array with shape ``(num_samples, 2)``,
                where column 0 is the left channel and column 1 is the right
                channel.

        Returns:
            A NumPy array with shape ``(num_samples, 2)``, where column 0 is
            the mid channel and column 1 is the side channel.
        """
        mid = (left_right[:, 0] + left_right[:, 1]) / 2
        side = (left_right[:, 0] - left_right[:, 1]) / 2
        return np.stack((mid, side), axis=1)

    @staticmethod
    def decode(mid_side: np.ndarray) -> np.ndarray:
        """Decode mid/side audio into left/right stereo format.

        Args:
            mid_side: Mid/side audio array with shape ``(num_samples, 2)``,
                where column 0 is the mid channel and column 1 is the side
                channel.

        Returns:
            A NumPy array with shape ``(num_samples, 2)``, where column 0 is
            the left channel and column 1 is the right channel.
        """
        left = (mid_side[:, 0] + mid_side[:, 1]) * 2
        right = (mid_side[:, 0] - mid_side[:, 1]) * 2
        return np.stack((left, right), axis=1)
