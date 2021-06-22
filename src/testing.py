from scipy import signal
import sounddevice as sd
import numpy as np
from scipy.fftpack import rfft, irfft
from time import time as now
from matplotlib import pyplot as plt
from correlation import forward_match

fs = 44100
f0 = 30
t1 = 100
f1 = 10000
plt.close('all')
t = np.arange(0, t1, 1/fs)

sig = signal.chirp(t, f0, t1, f1, method='quadratic',
                   phi=-90).astype(np.float32)
freq = f0 + (f1 - f0) * t**2 / t1**2
sig = np.pad(sig, (fs, fs))
received = sig[fs:-fs]*(0.8+0.2*np.cos(t*np.pi*2*2)).astype(np.float32)
# plt.plot(received[:fs*10:8])
# s = signal.chirp(t, 50, 100, 1000, 'logarithmic', -90).astype(np.float32)
# s1 = s[:-1000]
# s2 = s[1000:]
print(forward_match(sig, received, fs-1000, fs+1000, 1))
plt.plot(freq[:fs*10:8], np.abs(signal.hilbert(received))[:fs*10:8])
plt.show()
