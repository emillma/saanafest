from scipy.io import wavfile
import numpy as np
from correlation import get_patterns, forward_match
from utils import first_order_gain
from scipy.interpolate import interp1d


class Feeder:
    def __init__(self, channels=2, bsize=2048, fs=48000):
        self.bsize = bsize
        self.fs = fs
        self.tape_length = 4
        self.tape = np.zeros(
            (self.tape_length*fs, channels)).astype(np.float32)
        self.ramp = 1

        self.gain_pattern = np.ones(self.tape.shape[0])
        self.gain_pattern[:self.bsize] *= np.linspace(0, 1, self.bsize)

    def step(self, block, alpha=0.01, fixed_len=None):
        self.tape = np.pad(self.tape, ((0, self.bsize), (0, 0)))[self.bsize:]
        block = block.astype(np.float32)
        patterns, lengths = get_patterns(block, 100, 500, 4)
        for channel in range(block.shape[1]):
            pattern = patterns[:lengths[channel], channel]
            if fixed_len is not None:
                iterp = interp1d(np.linspace(0, 1, pattern.shape[0]), pattern)
                pattern = iterp(np.linspace(0, 1, fixed_len))
                pattern = pattern.astype(np.float32)

            shift = forward_match(
                self.tape[:pattern.shape[0]*2, channel], pattern,
                0, block.shape[0], 4)

            tiled = np.tile(pattern,
                            (self.tape.shape[0]-self.bsize)//pattern.shape[0])
            self.merge_in(shift, tiled, channel, alpha)
            pass
        return self.tape[:self.bsize]

    def merge_in(self, shift, tiled, channel, alpha):
        gain = self.gain_pattern[:tiled.shape[0]]*alpha
        self.tape[shift: shift+tiled.shape[0], channel] = (
            self.tape[shift: shift+tiled.shape[0], channel] * (1-gain)
            + tiled * gain)


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
