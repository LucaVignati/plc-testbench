import numpy as np
from enum import Enum

class CodecMode(Enum):
  ENCODE = 0
  DECODE = 1

class MidSideCodec(object):

  def __call__(self, audio: np.ndarray, type:CodecMode = CodecMode.ENCODE) -> np.ndarray:
    if type == CodecMode.ENCODE:
      return MidSideCodec.encode(audio)
    elif type == CodecMode.DECODE:
      return MidSideCodec.decode(audio)
    
  def encode(left_right: np.ndarray) -> np.ndarray:
    mid = (left_right[:,0] + left_right[:,1]) / 2
    side = (left_right[:,0] - left_right[:,1]) / 2
    return np.stack((mid, side), axis=1)

  def decode(mid_side: np.ndarray) -> np.ndarray:
    left = (mid_side[:,0] + mid_side[:,1]) * 2
    right = (mid_side[:,0] - mid_side[:,1]) * 2
    return np.stack((left, right), axis=1)