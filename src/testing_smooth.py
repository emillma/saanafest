import sounddevice as sd
import numpy as np
from feeder import Feeder
from fightertwister.src.fightertwister.utils import ft_colors
from ftbro import FtBro, to_range
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
        self.fs = 44100
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
            print((self.tlog[-1][1]-self.tlog[-1][0])/(self.bsize/self.fs))
            self.ft.save()
            self.ft.do_task_delay(1000, foo)
        self.foo = foo

        for node in self.ft.nodes:
            params = node.get_property('params')
            params[1, 0].set_value(1)
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
            np.sort(np.ravel(np.array([[60, 90]]).T*np.arange(1, 10, 2)))]
        self.current_tones = [i[0] for i in self.valid_tones]
        self.shift_rate = 1
        self.sine_times = np.zeros(self.node_channels_out)

        self.tlog = [[time.time(), time.time()]*2]
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

        current_tones = self.get_current_tones()

        sound = self.feeder.step(indata, alpha, current_tones)
        sound *= 10
        sound = self.handle_controlled_sine(sound)

        gain = (self.ft.nodes.value.ravel()[None, :sound.shape[1]]
                * self.ft.main_volume.value).astype(np.float32)
        sound *= gain
        outdata[:] = sound

        self.show_output_volue(outdata)
        self.tlog.append([t0, time.time()])
        self.tlog.pop(0)

    def get_current_tones(self):
        for i in range(self.node_channels_out):
            shift_mode, shift_value = self.ft.get_shift_rate(i)
            if shift_mode == 0:
                rate = to_range(shift_value, 0.1, 10)
                if np.random.random() < 1/(rate*self.fs/self.bsize):
                    self.current_tones[i] = np.random.choice(
                        self.valid_tones[i])
                    self.ft.encoder_slots[0, 2, 3].flash_color(ft_colors.blue)
            else:
                self.current_tones[i] = self.valid_tones[i][
                    round(to_range(shift_value, len(self.valid_tones[i])-1, 0))]
        return self.current_tones

    def handle_controlled_sine(self, sound):
        for i in range(self.node_channels_out):
            shift_mode, shift_value = self.ft.get_contolled_tone(i)
            if shift_mode == 0:
                continue
            else:
                freq = 50+np.exp(shift_value*np.log(9950))
                end_time = (self.bsize/self.fs) * 2*np.pi*freq
                t = np.linspace(0, end_time, self.bsize) + self.sine_times[i]
                generator = [np.sin, signal.sawtooth,
                             signal.square][shift_mode-1]
                signal_gain = [1, 2/3, 1 / 2][shift_mode-1]
                sine = generator(t)*signal_gain
                self.sine_times[i] += end_time * (self.bsize+1)/self.bsize
                sound[:, i] = sine
        return sound

    def cb_mic(self, indata, _frames, _time, _status):
        # print(indata.shape)
        self.mic_slot.set(np.random.random(indata.shape))

    def show_input_volue(self, indata):
        inputs = self.moniotors[:2, :3].ravel()
        inputs[:indata.shape[1]].show_value(np.amax(indata, axis=0))

    def show_output_volue(self, outdata):
        outputs = self.moniotors[2:, :3].ravel()
        outputs[:outdata.shape[1]].show_value(np.amax(outdata, axis=0))

    def run(self):
        with self.ft, self.sd_stream_nodes:
            self.ft.do_task_delay(500, self.foo)
            input('enter to quit')


if __name__ == '__main__':
    bro = Bro()
    bro.run()
