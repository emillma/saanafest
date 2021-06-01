from scipy import signal
import sounddevice as sd
import numpy as np
from feedback import Feeder
from matplotlib import pyplot as plt
from correlation import forward_match
from utils import first_order_gain

fs = 48000
BSIZE = 2048
CHANNELS = 1
emil = Feeder(CHANNELS, BSIZE, fs)


class Chirp:
    def __init__(self):
        duration = 60
        times = np.linspace(0, duration, fs*duration)
        self.chirp = signal.chirp(
            times, 50, duration, 440*4, 'logarithmic', -90)
        self.chirp = np.pad(self.chirp, fs)

        self.i = 0
        self.done = False
        self.record = np.stack(3*[np.zeros_like(self.chirp)], axis=-1)

    def get_sample(self, indata):
        if self.i+BSIZE >= self.chirp.shape[0]:
            self.done = True
            return np.zeros((BSIZE, 1), np.float32)
        else:
            out = self.chirp[self.i:self.i+BSIZE]
            self.record[self.i:self.i+BSIZE] = indata
            self.i += BSIZE
            return out[:, None]


chirp = Chirp()


def callback(indata, outdata, frames, time, status) -> None:
    contact_in = indata[:, 0:1]
    mic_in = indata[:, 1:2]

    outdata[:] = chirp.get_sample(indata)
    # print(outdata.shape)
    print(f'{np.amax(contact_in):.3f}',
          f'{np.amax(mic_in):.3f}',
          f'{np.amax(outdata):.3f}')


audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'
yeti_in = 'Microphone (Yeti Stereo Microph, MME'
pc_out = 'Speakers (Realtek High Definiti, MME'
sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'

with sd.Stream(samplerate=fs,
               blocksize=BSIZE,
               device=(audiobox_in, audiobox_out),
               channels=(3, CHANNELS),
               dtype=np.float32,
               latency='low',
               callback=callback
               ):
    while not chirp.done:
        sd.sleep(1)
# input('enter to quit')
# plt.plot(np.abs(signal.hilbert(chirp.record[:, 1])))
for i in range(3):
    freqs, times, spec = signal.spectrogram(chirp.record[:, i], fs, nperseg=1024*8,
                                            noverlap=1024*7)

    plt.plot(freqs, 10*np.log10(np.amax(spec, axis=1)))
plt.show()
