import numpy as np


def first_order_gain(length, peak, sustain, up_const, down_const):
    gain = np.arange(length, dtype=np.float)
    gain[:peak] = (1-np.exp(-gain[:peak]/up_const))/(1-np.exp(-peak/up_const))
    gain[peak:sustain] = 1
    gain[sustain:] = ((np.exp((-(gain[sustain:]-sustain))/down_const)
                       - np.exp(-(length-sustain)/down_const))
                      / (1-np.exp(-(length-sustain)/down_const)))
    return gain
