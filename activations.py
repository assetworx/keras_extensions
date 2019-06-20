import numpy as np
from .utils import sigmoid

#
# SWISH ACTIVATION FUNCTION
# @rationale
# @url https://arxiv.org/pdf/1710.05941.pdf
#
def swish(x):
  return x * sigmoid(x)

#
# E-SWISH ACTIVATION FUNCTION
# @rationale
# @url https://arxiv.org/pdf/1801.07145.pdf
def eswish(x, B = 1):
  return B * x * sigmoid(x)

#
# SINERELU ACTIVATION FUNCTION
# @rationale
# @url https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d
def sinerelu(x, epsilon = 0.025):
  return x if x > 0 else epsilon*(np.sin(x) - np.cos(x))

# LEAKY RELU
# @rationale
# @url https://arxiv.org/pdf/1505.00853.pdf
# @note since Keras applies the [0, ∞) range instead of [1, ∞), we allow it too.
def leakyrelu(x, alpha = 0.3):
  if alpha < 0:
    raise Exception('Alpha not within accepted range.')
  return x if x >= 0 else x/alpha