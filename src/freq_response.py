from scipy import signal
import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt
from correlation import forward_match
import time


audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'
sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'

asio = 'ASIO4ALL v2, ASIO'


class Chirp:
    def __init__(self, node):
        self.fs = 44100
        self.f0 = 40
        self.t1 = 60
        self.f1 = 1000
        self.bsize = 2048

        self.times = np.arange(0, self.t1, 1/self.fs)
        self.chirp = signal.chirp(self.times, self.f0, self.t1, self.f1,
                                  method='quadratic',
                                  phi=-90).astype(np.float32)
        self.freq = self.f0 + (self.f1 - self.f0) * self.times**2 / self.t1**2

        self.chirp_padded = np.pad(self.chirp, self.fs)

        self.chirp_idx = 0
        self.done = False
        self.record = np.zeros_like(self.chirp_padded)

        ca_in = sd.AsioSettings(channel_selectors=[0])
        ca_out = sd.AsioSettings(channel_selectors=[0])

        self.stream = sd.Stream(samplerate=self.fs,
                                blocksize=self.bsize,
                                device=asio,
                                channels=(1, 1),
                                dtype=np.float32,
                                latency='low',
                                # extra_settings=(ca_in, ca_out),
                                callback=self.callback)

    def get_sample(self, indata):
        if self.chirp_idx+self.bsize >= self.chirp_padded.shape[0]:
            self.done = True
            return np.zeros((self.bsize, 1), np.float32)
        else:
            out = self.chirp_padded[self.chirp_idx:self.chirp_idx+self.bsize]
            self.record[self.chirp_idx:self.chirp_idx + self.bsize
                        ] = indata.ravel()
            self.chirp_idx += self.bsize
            return out[:, None]

    def callback(self, indata, outdata, frames, time, status) -> None:
        outdata[:] = self.get_sample(indata)

    def run(self):
        with self.stream:
            while not self.done:
                time.sleep(0.1)
        filter = signal.butter(3, (self.f0, self.f1), 'bandpass', fs=self.fs)
        filtered = signal.filtfilt(*filter, self.record).astype(np.float32)
        shift = -forward_match(self.chirp_padded, filtered,
                               -self.fs, 0, 4)
        print(shift)
        analytic_signal = signal.hilbert(
            filtered[self.fs+shift:-self.fs+shift])

        instantaneous_amp = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0*np.pi) * self.fs)
        response_y = instantaneous_amp
        response_x = self.freq
        plt.plot(response_x[::100], response_y[::100])
        # plt.plot(instantaneous_frequency)
        # plt.plot(self.record[self.fs-100: self.fs+self.fs])
        plt.show()


chirp = Chirp(1)
chirp.run()


# plt.close('all')
# t = np.arange(0, 100, 1/48000)
# signal.chirp(np.arange(10000), 0.01, 10000, 0.1),
# s = signal.chirp(t, 50, 100, 1000, 'logarithmic', -90).astype(np.float32)
# s1 = s[:-1000]
# s2 = s[1000:]
# print(forward_match(s1, s2, 1, 2000, 4))
# plt.plot(np.abs(signal.hilbert(s))[::1000])
