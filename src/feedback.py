from scipy.signal import chirp, hilbert
from scipy.io import wavfile
from correlation import get_patterns
import numpy as np
from correlation import get_patterns, forward_match
from utils import first_order_gain


class Feeder:
    def __init__(self, channels=2, bsize=2048, fs=48000):
        self.bsize = bsize
        self.fs = fs
        self.tape_length = fs*4
        self.tape = np.zeros((self.tape_length, channels)).astype(np.float32)
        self.ramp = 1
        self.gain = first_order_gain(self.tape_length,
                                     bsize*2, fs,
                                     bsize, fs)
        # self.gain = np.ones(self.tape_length)

    def step(self, block, alpha=0.01):
        self.tape = np.pad(self.tape, ((0, self.bsize), (0, 0)))[self.bsize:]
        block = block.astype(np.float32)
        patterns, lengths = get_patterns(block, 100, 500, 4)
        for channel in range(block.shape[1]):
            pattern = patterns[:lengths[channel], channel]
            shift = forward_match(
                self.tape[:pattern.shape[0]*2, channel], pattern,
                0, block.shape[0], 4)

            tiled = np.tile(pattern,
                            (self.tape.shape[0]-self.bsize)//pattern.shape[0])
            tiled = (tiled * self.gain[:tiled.shape[0]])
            self.merge_in(shift, tiled, channel, alpha)
            pass
        return self.tape[:self.bsize]

    def merge_in(self, shift, tiled, channel, alpha):
        self.tape[shift: shift+tiled.shape[0], channel] = (
            (1-alpha) * self.tape[shift:shift + tiled.shape[0], channel]
            + alpha * tiled
        )


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
        parts.append(emil.step(block))
    out = np.vstack(parts)
    out /= np.amax(np.abs(out))
    print(time.time()-t0)
    sd.play(out, fs, blocking=True)
    pass
