from keras import backend as K

#
# SWISH ACTIVATION FUNCTION
# @url https://arxiv.org/pdf/1710.05941.pdf
#
def swish(x):
  return x * K.sigmoid(x)