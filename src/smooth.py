from numpy.core.fromnumeric import argmin
from scipy.io import wavfile
import sounddevice as sd
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

plt.close('all')
fs, data = wavfile.read('cello.wav')
# data = data[10*fs:]
# window = 1024
# window_h = window//2
# out = np.zeros(700000, float)

# time_const = 0.9
# for i in range(550):
#     sample = data[window*i:window*(i+1), 0].astype(np.int64)
#     start = window*i
#     assert(np.any(sample))
#     correlation = np.correlate(
#         np.concatenate([sample[100:], np.zeros(800)]), sample,
#         mode='valid')
#     # plt.plot(correlation)
#     # plt.pause(0.2)
#     # plt.clf()
#     best_fit = np.argmax(correlation)+100
#     emil = sample[:-(sample.size % best_fit) or None].reshape(-1, best_fit)

#     tone = np.mean(emil, axis=0)
#     incorr = np.correlate(out[start:start+2*tone.shape[0]], tone, 'valid')
#     inshitf = np.argmax(incorr)
#     n = 100 * window // best_fit
#     tmp = 0.02 * np.tile(tone, n) * np.sin(np.linspace(0, np.pi, best_fit*n))
#     out[start+inshitf:start+inshitf+best_fit*n] += tmp
#     assert not np.any(np.isnan(tone))


# # b, a = signal.butter(2, 4000, 'low', fs=48000, output='ba')
# # out = signal.filtfilt(b, a, out)
# # out = out / np.amax(np.abs(out))

# wavfile.write('smooth.wav', fs, out)
print(data.shape)
sd.play(data, fs, blocking=True)
