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

        self.node_gains = np.exp(-0.01*np.arange(self.tape_length//self.bsize,
                                                 0, -1)).astype(np.float32)
        self.mic_gains = self.node_gains.copy()

        self.node_gains_repeated = np.empty(self.tape_length, np.float32)
        self.mic_gains_repeated = self.node_gains_repeated.copy()
        self.ramp = 1

        self.gain_ramp_up = np.linspace(0, 1, self.bsize).astype(np.float32)

    def roll_tape(self):
        self.tape = np.roll(self.tape, -self.bsize, axis=0)
        self.tape[-self.bsize:] = 0
        self.node_gains = np.roll(self.node_gains, -1, axis=0)
        self.node_gains[-1] = 1

    def step_node(self, block, alpha=0.01, fixed_lengts=None):
        block = block.astype(np.float32)
        patterns, lengths = get_patterns(block, 44, 1000, 8)

        self.node_gains *= (1-alpha)
        self.node_gains_repeated[:] = np.repeat(alpha/self.node_gains,
                                                self.bsize)

        for channel in range(block.shape[1]):
            pattern_len = lengths[channel]
            pattern = patterns[:pattern_len, channel]

            if fixed_lengts is not None:
                fl = fixed_lengts[channel]
                fl_closest = fl[np.argmin(
                    np.abs(np.log2(fl)-np.log2(pattern_len)))]
                iterp = interp1d(np.linspace(0, 1, pattern_len), pattern,
                                 assume_sorted=True)
                pattern = iterp(np.linspace(0, 1, fl_closest))
                pattern = pattern.astype(np.float32)

            self.merge_in(pattern, pattern_len, channel,
                          self.node_gains_repeated)
        return self.tape[:self.bsize]

    def step_mic(self, block, alpha=0.01, fixed_lengts=None):
        assert block.shape[1] == 1
        block = block.astype(np.float32)
        patterns, lengths = get_patterns(block, 44, 1000, 8)

        self.mic_gains *= (1-alpha)
        self.mic_gains_repeated[:] = np.repeat(alpha/self.mic_gains,
                                               self.bsize)
        pattern_len = lengths[0]
        pattern = patterns[:pattern_len, 0]

        if fixed_lengts is not None:
            best_diff = np.inf
            for channel in range(self.tape.shape[1]):
                fl = fixed_lengts[channel]
                fl_closest = fl[np.argmin(
                    np.abs(np.log2(fl)-np.log2(pattern_len)))]

        iterp = interp1d(np.linspace(0, 1, pattern_len), pattern,
                         assume_sorted=True)
        pattern = iterp(np.linspace(0, 1, fl_closest))
        pattern = pattern.astype(np.float32)

        self.merge_in(pattern, pattern_len, channel,
                      self.mic_gains_repeated)
        return self.tape[:self.bsize]

    def merge_in(self, pattern, pattern_len, channel, gains_repeated):
        shift = forward_match(
            self.tape[:pattern_len*2, channel], pattern,
            0, self.bsize, 8)
        tiled = np.tile(pattern,
                        (self.tape.shape[0]-self.bsize)//pattern.shape[0])
        tiled[:self.bsize] *= self.gain_ramp_up
        tiled[:] *= gains_repeated[:tiled.shape[0]]
        self.tape[shift: shift+tiled.shape[0], channel] += tiled
        self.tape[:self.bsize, channel] *= gains_repeated[0]
        return self.tape[:self.bsize]


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
        parts.append(emil.step_node(block, alpha=0.001))
    out = np.vstack(parts)
    out /= np.amax(np.abs(out))
    print(time.time()-t0)
    sd.play(out, fs, blocking=True)
    pass
