import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from itertools import repeat

input_device = 'Line (MCHStreamer TDM8)'
output_device = 'Speakers (MCHStreamer TDM8)'

fig, ax = plt.subplots(2, 1)
signal = np.stack([np.cos(np.linspace(0, 2*np.pi*i, 2048))
                   for i in range(1, 9)], -1)

indata_storage = signal.copy()
ax[0].plot(signal)
lines = ax[1].plot(indata_storage)


def cb_anim(i):
    for i in range(8):
        lines[i].set_data(np.arange(2048), indata_storage[:, i])
    return lines


anim = FuncAnimation(fig, cb_anim, repeat([1]), interval=100, blit=True)
plt.show(block=False)


def callback(indata, outdata, _frames, _time, _status):
    outdata[:] = signal+np.random.random(signal.shape)*0
    indata_storage[:] = indata


sd_stream = sd.Stream(samplerate=48000,
                      blocksize=2048,
                      device=(input_device,
                              output_device),
                      channels=(8,
                                8),
                      dtype=np.float32,
                      latency='low',
                      callback=callback)

with sd_stream:
    plt.show(block=True)
print('done')
