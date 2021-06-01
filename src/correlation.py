from matplotlib import pyplot as plt
import numba as nb
import numpy as np
from numpy.core.numeric import zeros_like


@nb.njit(['int32(int32[:], int32[:], int64, int64)',
          'float32(float32[:], float32[:], int64, int64)'],
         cache=True)
def correlate_single(arr0, arr1, shift, step=1):
    """move arr1 shift relative to arr0"""

    if shift >= 0:
        overlap = min(arr0.shape[0] - shift, arr1.shape[0])
        if overlap <= 0:
            return np.int32(0)
        return np.sum(arr0[shift:shift+overlap:step] * arr1[:overlap:step])
    else:
        overlap = min(arr0.shape[0], arr1.shape[0] + shift)
        if overlap <= 0:
            return np.int32(0)
        return np.sum(arr0[:overlap:step] * arr1[-shift:-shift+overlap:step])


@nb.njit(['int64(int32[:], int32[:], int64, int64, int64)',
          'int64(float32[:], float32[:], int64, int64, int64)'],
         cache=True, parallel=False)
def forward_match(arr0, arr1, min_shift, max_shift, rough_step):
    correlation_vals = np.zeros(max_shift - min_shift, arr0.dtype)

    for i in nb.prange((max_shift - min_shift)//rough_step):
        correlation_vals[i*rough_step] = correlate_single(
            arr0, arr1,
            min_shift + i*rough_step,
            step=rough_step)
    best_arg = np.argmax(correlation_vals)
    if rough_step != 1:
        for extra_step in nb.prange(-rough_step, rough_step+1):
            if 0 < best_arg + extra_step < correlation_vals.shape[0]:
                correlation_vals[best_arg + extra_step] = correlate_single(
                    arr0, arr1,
                    min_shift + best_arg + extra_step,
                    1)

        best_arg = np.argmax(correlation_vals)

    return best_arg + min_shift


@nb.njit(['int32[:](int32[:], int64, int64, int64)',
          'float32[:](float32[:], int64, int64, int64)'], cache=True)
def get_repeating_pattern(arr, min_shift, max_shift, rough_step):
    best_arg = forward_match(arr, arr, min_shift, max_shift, rough_step)
    output = np.zeros(best_arg, arr.dtype)
    arr_cropped = arr[0:(arr.shape[0] // best_arg) * best_arg]
    for i in nb.prange(best_arg):
        output[i] = np.mean(arr_cropped[i::best_arg])

    diff = (output[0] - output[-1]) - (output[1] - output[-2])/3
    for i in range(output.shape[0]):
        output[i] += (i*diff)/(output.shape[0]) - diff/2

    return output


@nb.njit(['Tuple((int32[:,:], int64[:]))(int32[:,:], int64, int64, int64)',
          'Tuple((float32[:,:], int64[:]))(float32[:,:], int64, int64, int64)'],
         cache=True)
def get_patterns(arr, min_shift, max_shift, rough_step):
    patterns = np.zeros((max_shift, arr.shape[1]), arr.dtype)
    lengths = np.zeros(arr.shape[1], np.int64)
    for i in range(arr.shape[1]):
        pattern = get_repeating_pattern(
            arr[:, i], min_shift, max_shift,  rough_step)
        patterns[:pattern.shape[0], i] = get_repeating_pattern(
            arr[:, i], min_shift, max_shift,  rough_step)
        lengths[i] = pattern.shape[0]
    return (patterns, lengths)
