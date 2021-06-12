import sounddevice as sd
import numpy as np
from feeder import Feeder
from ftbridge import FtBridge
import time
from soundslot import SoundSlot


audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'

mch_out = 'Speakers (MCHStreamer TDM8)'
mch_in = 'Line (MCHStreamer TDM8)'

yeti_in = 'Microphone (Yeti Stereo Microph, MME'

pc_in = 'Microphone Array (Realtek High '
pc_out = 'Speakers (Realtek High Definiti, MME'

sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'


class Bro:
    def __init__(self):
        self.fs = 48000
        self.bsize = 2048
        self.mic_device_in = mch_in
        self.mic_channels_in = 1

        self.node_device_in = pc_in
        self.node_channels_in = 1
        self.node_device_out = sonywh_out
        self.node_channels_out = 1

        self.feeder = Feeder(self.node_channels_out, self.bsize)
        self.ft = FtBridge()

        self.mic_slot = SoundSlot(
            np.zeros((self.bsize, self.mic_channels_in), np.float32))

        self.moniotors = self.ft.encoders[1]
        self.moniotors.set_follow_value(0)

        self.sd_stream_mic = sd.InputStream(
            samplerate=self.fs,
            blocksize=self.bsize,
            device=self.mic_device_in,
            channels=self.mic_channels_in,
            dtype=np.float32,
            latency='low',
            callback=self.cb_mic)

        self.sd_stream_nodes = sd.Stream(
            samplerate=self.fs,
            blocksize=self.bsize,
            device=(self.node_device_in,
                    self.node_device_out),
            channels=(self.node_channels_in,
                      self.node_channels_out),
            dtype=np.float32,
            latency='low',
            callback=self.cb_node)
        self.fixed_len = 400

    def cb_node(self, indata, outdata, _frames, _time, _status):
        t0 = time.time()
        self.show_input_volue(indata)
        if np.amax(indata) < 0.01:
            alpha = 0.0001
        else:
            alpha = 0.01

        self.fixed_len = 300

        sound = self.feeder.step(indata, alpha=alpha)

        sound = sound * self.ft.main_volume.value * 10
        outdata[:] = sound + 0

        self.show_output_volue(outdata)

    def cb_mic(self, indata, _frames, _time, _status):
        # print(indata.shape)
        self.mic_slot.set(np.random.random(indata.shape))

    def show_input_volue(self, indata):
        inputs = self.moniotors[:2, :3].ravel()
        inputs[:indata.shape[1]]._show_value(np.amax(indata, axis=0))

    def show_output_volue(self, outdata):
        outputs = self.moniotors[2:, :3].ravel()
        outputs[:outdata.shape[1]]._show_value(np.amax(outdata, axis=0))

    def run(self):
        with self.ft, self.sd_stream_nodes, self.sd_stream_mic:
            input('enter to quit')


if __name__ == '__main__':
    bro = Bro()
    bro.run()
