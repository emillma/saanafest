import sounddevice as sd
import numpy as np
from feeder import Feeder
from ftbridge import FtBridge
import time

audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'
yeti_in = 'Microphone (Yeti Stereo Microph, MME'
pc_out = 'Speakers (Realtek High Definiti, MME'
sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'


class Bro:
    def __init__(self):
        self.fs = 48000
        self.bsize = 2048
        self.device_in = None
        self.device_out = None
        self.channels_out = 1
        self.channels_in = 1
        self.feeder = Feeder(self.channels_out, self.bsize)
        self.ft = FtBridge()

        self.moniotors = self.ft.encoders[1]
        self.moniotors.set_follow_value(0)

        self.sd_stream = sd.Stream(samplerate=self.fs,
                                   blocksize=self.bsize,
                                   device=(self.device_in,
                                           self.device_out),
                                   channels=(self.channels_in,
                                             self.channels_out),
                                   dtype=np.float32,
                                   latency='low',
                                   callback=self.cb_sd)
        self.fixed_len = 400

    def cb_sd(self, indata, outdata, _frames, _time, _status):
        t0 = time.time()
        self.show_input_volue(indata)
        if np.amax(indata) < 0.01:
            alpha = 0.0001
        else:
            alpha = 0.01

        self.fixed_len = 300

        sound = self.feeder.step(indata, alpha=alpha, fixed_len=self.fixed_len)

        sound = sound * self.ft.main_volume.value * 10
        outdata[:] = sound + 0

        self.show_output_volue(outdata)

    def show_input_volue(self, indata):
        inputs = self.moniotors[:2, :3].ravel()
        inputs[:indata.shape[1]]._show_value(np.amax(indata, axis=0))

    def show_output_volue(self, outdata):
        outputs = self.moniotors[2:, :3].ravel()
        outputs[:outdata.shape[1]]._show_value(np.amax(outdata, axis=0))

    def run(self):
        with self.ft, self.sd_stream:
            input('enter to quit')


if __name__ == '__main__':
    bro = Bro()
    bro.run()
