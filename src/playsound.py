from scipy import signal
import sounddevice as sd
import numpy as np
from feedback import Feeder
from matplotlib import pyplot as plt
from correlation import forward_match
from utils import first_order_gain
import soundfile as sf
from pathlib import Path
BSIZE = 2048
CHANNELS = 1
song, fs = sf.read(Path(__file__).parents[1].joinpath('cello.wav'))
emil = Feeder(CHANNELS, BSIZE, fs)


class Song:
    def __init__(self):
        self.song = song
        self.i = 0
        self.done = False

    def get_sample(self, indata):
        if self.i+BSIZE >= self.song.shape[0]:
            self.done = True
            return np.zeros((BSIZE, 1), np.float32)
        else:
            out = self.song[self.i:self.i+BSIZE, 0]
            self.i += BSIZE
            return out[:, None]


song = Song()


def callback(indata, outdata, frames, time, status) -> None:
    contact_in = indata[:, 0:1]
    mic_in = indata[:, 1:2]
    # songblock = song.get_sample(indata)
    outdata[:] = emil.step(mic_in)*10
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
               device=(audiobox_in, sonywh_out),
               channels=(2, CHANNELS),
               dtype=np.float32,
               latency='low',
               callback=callback
               ):
    while not song.done or True:
        sd.sleep(1)
