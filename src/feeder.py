from scipy.io import wavfile
import numpy as np
from correlation import get_patterns, forward_match
from scipy.interpolate import interp1d


class Feeder:
    def __init__(self, channels=2, bsize=2048, fs=48000):
        self.bsize = bsize
        self.fs = fs
        self.tape_length = (4*fs//self.bsize)*self.bsize
        self.tape = np.zeros(
            (self.tape_length, channels)).astype(np.float32)
        self.gains = np.ones(self.tape_length//self.bsize, np.float32)
        self.gains = np.exp(-0.01*np.arange(self.gains.size, 0, -1))
        self.ramp = 1

        self.gain_ramp_up = np.linspace(0, 1, self.bsize).astype(np.float32)

    def step(self, block, alpha=0.01, fixed_lengts=None):
        self.tape = np.roll(self.tape, -self.bsize, axis=0)
        self.tape[-self.bsize:] = 0
        self.gains = np.roll(self.gains, -1, axis=0)
        self.gains[-1] = 1

        block = block.astype(np.float32)
        patterns, lengths = get_patterns(block, 44, 1000, 8)
        for channel in range(block.shape[1]):
            patten_len = lengths[channel]
            pattern = patterns[:patten_len, channel]
            if fixed_lengts is not None:
                fl = fixed_lengts[channel]
                if isinstance(fl, np.ndarray):
                    fl_closest = fl[np.argmin(
                        np.abs(np.log2(fl)-np.log2(patten_len)))]
                else:
                    fl_closest = fl
                # TODO if closer than threashold, dont shift
                iterp = interp1d(np.linspace(0, 1, patten_len), pattern,
                                 assume_sorted=True)
                pattern = iterp(np.linspace(0, 1, fl_closest))
                pattern = pattern.astype(np.float32)

            shift = forward_match(
                self.tape[:patten_len*2, channel], pattern,
                0, block.shape[0], 8)

            tiled = np.tile(pattern,
                            (self.tape.shape[0]-self.bsize)//pattern.shape[0])
            self.merge_in(shift, tiled, channel, alpha)
            pass
        return self.tape[:self.bsize]

    def merge_in(self, shift, tiled, channel, alpha):
        self.gains[:] *= (1-alpha)
        tiled[:self.bsize] *= self.gain_ramp_up
        tiled[:] *= (alpha
                     / np.repeat(self.gains, self.bsize)[:tiled.shape[0]])
        self.tape[shift: shift+tiled.shape[0], channel] += tiled
        self.tape[:self.bsize] *= self.gains[0]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sounddevice as sd
    import time
    plt.close('all')
    fs, data = wavfile.read('cello.wav')
    data = (data/np.amax(data)).astype(np.float32)
    bsize = 2048
    emil = Feeder(2, bsize, fs)
    parts = []
    t0 = time.time()
    for i in range(0, data.shape[0], bsize):
        block = data[i:i+bsize]
        parts.append(emil.step(block, alpha=0.001))
    out = np.vstack(parts)
    out /= np.amax(np.abs(out))
    print(time.time()-t0)
    sd.play(out, fs, blocking=True)
    pass
