import sounddevice as sd
import numpy as np
from feeder import Feeder
from fightertwister import FtBro
import time
from soundslot import SoundSlot
from scipy import signal

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
        self.ft = FtBro()

        self.mic_slot = SoundSlot(
            np.zeros((self.bsize, self.mic_channels_in), np.float32))

        self.moniotors = self.ft.encoders[1]
        self.moniotors.set_follow_value(0)

        self.filter = signal.butter(3, (50, 150), 'bandpass', fs=self.fs)
        self.zi = np.stack([signal.lfilter_zi(*self.filter)], axis=-1)

        self.start_time = time.time()

        def foo():
            print(self.current_tones)
            self.ft.do_task_delay(500, foo)

        self.foo = foo
        # self.sd_stream_mic = sd.InputStream(
        #     samplerate=self.fs,
        #     blocksize=self.bsize,
        #     device=self.mic_device_in,
        #     channels=self.mic_channels_in,
        #     dtype=np.float32,
        #     latency='low',
        #     callback=self.cb_mic)

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

        self.valid_tones = self.node_channels_in*[
            np.ravel(np.array([[60, 90]]).T*np.arange(1, 10, 2))]
        self.current_tones = [i[0] for i in self.valid_tones]
        self.shift_rate = 10

        self.cb_node(
            np.empty((self.bsize, self.node_channels_in), np.float32),
            np.empty((self.bsize, self.node_channels_out), np.float32),
            None,
            None,
            None
        )

    def cb_node(self, indata, outdata, _frames, _time, _status):
        t0 = time.time()
        for channel in range(self.node_channels_in):
            low = self.ft.get_node_values(channel)[1][0, 0]
            high = self.ft.get_node_values(channel)[1][1, 0]
            low = 20 + np.exp(low*np.log(10000))
            high = max(low+100, 20 + np.exp(high*np.log(10000)))
            indata, self.zi[:] = signal.lfilter(
                *signal.butter(3, (low, high), 'bandpass', fs=self.fs),
                indata,
                axis=0, zi=self.zi)
        self.show_input_volue(indata)
        if np.amax(indata) < 0.01:
            alpha = 0.001
        else:
            alpha = 0.01

        self.fixed_len = 300

        for i in range(len(self.current_tones)):
            if np.random.random() < 1/(self.shift_rate*self.fs/self.bsize):
                self.current_tones[i] = np.random.choice(self.valid_tones[i])

        sound = self.feeder.step(indata, alpha, self.current_tones)

        gain = (10 * self.ft.nodes.value.ravel()[None, :sound.shape[1]]
                * self.ft.main_volume.value).astype(np.float32)
        sound *= gain
        outdata[:] = indata

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
        with self.ft, self.sd_stream_nodes:
            self.ft.do_task_delay(500, self.foo)
            input('enter to quit')


if __name__ == '__main__':
    bro = Bro()
    bro.run()
