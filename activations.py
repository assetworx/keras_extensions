from keras import backend as K

#
# SWISH ACTIVATION FUNCTION
# @rationale
# @url https://arxiv.org/pdf/1710.05941.pdf
#
def swish(x):
  return x * K.sigmoid(x)

#
# E-SWISH ACTIVATION FUNCTION
# @rationale
# @url https://arxiv.org/pdf/1801.07145.pdf
def eswish(x, B = 1):
  return B * x * K.sigmoid(x)