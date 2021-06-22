from numpy.core.fromnumeric import mean
import sounddevice as sd
import numpy as np
from feeder import Feeder
from fightertwister.src.fightertwister.utils import ft_colors
from ftbro import FtBro, to_range
import time
from soundslot import SoundSlot
from scipy import signal
import logging

audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'

mch_out = 'Speakers (MCHStreamer TDM8)'
mch_in = 'Line (MCHStreamer TDM8)'

yeti_in = 'Microphone (Yeti Stereo Microph, MME'

pc_in = 'Microphone Array (Realtek High '
pc_out = 'Speakers (Realtek High Definiti, MME'

sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'
asio = 'ASIO4ALL v2, ASIO'


class Bro:
    def __init__(self):
        self.fs = 44100
        self.bsize = 2048
        self.mic_device_in = mch_in
        self.mic_channels_in = 1

        self.node_device_in = asio
        self.node_n_channels = 4
        self.node_device_out = asio

        self.feeder = Feeder(self.node_n_channels, self.bsize)
        self.ft = FtBro()

        self.mic_slot = SoundSlot(
            np.zeros((self.bsize, self.mic_channels_in), np.float32))

        self.moniotors = self.ft.encoders[1]
        self.moniotors.set_follow_value(0)

        self.filter = signal.butter(3, (50, 2000), 'bandpass', fs=self.fs)
        self.zi = np.stack([signal.lfilter_zi(*self.filter)]
                           * self.node_n_channels, axis=-1)

        self.start_time = time.time()
        self.freq_shift_times = np.zeros(self.node_n_channels)

        def repeat_foo():
            # print((self.tlog[-1][1]-self.tlog[-1][0])/(self.bsize/self.fs))
            self.ft.save()
            # print(self.sd_stream_nodes.cpu_load)
            self.ft.do_task_delay(5000, repeat_foo)
        self.repeat_foo = repeat_foo

        # self.sd_stream_mic = sd.InputStream(
        #     samplerate=self.fs,
        #     blocksize=self.bsize,
        #     device=self.mic_device_in,
        #     channels=self.mic_channels_in,
        #     dtype=np.float32,
        #     latency='low',
        #     callback=self.cb_mic)

        ca_in = sd.AsioSettings(channel_selectors=[1, 0])
        ca_out = sd.AsioSettings(channel_selectors=[0, 1])

        self.sd_stream_nodes = sd.Stream(
            samplerate=self.fs,
            blocksize=self.bsize,
            device=(self.node_device_in,
                    self.node_device_out),
            channels=(self.node_n_channels+1,
                      self.node_n_channels),
            dtype=np.float32,
            latency='low',
            # extra_settings=(ca_in, ca_out),
            callback=self.cb_node)

        self.valid_tones = [np.array(tones) for tones in [
            [128.6, 179.6, 215.2, 252.4, 271.4, 296, 380, 502.5, 562.2, 598],
            [320, 160],
            [320, 160],
            [320, 160],
        ]][:self.node_n_channels]
        self.valid_lengths = [np.round(self.fs/tones).astype(int)
                              for tones in self.valid_tones]
        self.current_lengths = [i[0:1] for i in self.valid_lengths]

        self.shift_rate = 1
        self.sine_times = np.zeros(self.node_n_channels)

        self.tlog = [[time.time(), time.time()]*2]
        # self.cb_node(
        #     np.zeros(
        #         (self.bsize, self.node_n_channels+1), np.float32),
        #     np.empty((self.bsize, self.node_n_channels), np.float32),
        #     None,
        #     time.time(),
        #     None
        # )

    def cb_node(self, indata, outdata, _frames, _time, _status):
        t0 = time.time()
        # indata[:] = np.sin(np.linspace(
        #     0, self.bsize/self.fs, self.bsize)*2*np.pi*400)[:, None]
        mean_squared = np.sqrt(2 * np.mean(indata**2, axis=0)[None, :])
        indata[:] = np.where(mean_squared > 0.01,
                             indata/mean_squared, 0)

        input_node = indata[:, :self.node_n_channels]
        input_mic = indata[:, -1:]
        input_node *= min(1, (1-self.ft.mic_volume.value)*2)
        input_mic *= min(1, self.ft.mic_volume.value*2)
        self.show_input_volue(mean_squared)

        current_tones = self.get_current_tones(_time.currentTime)
        # current_tones = None
        sound = self.feeder.step_node(
            input_node, alpha=0.05, fixed_lengts=current_tones)
        sound, best_node, best_len = self.feeder.step_mic(
            input_mic, alpha=0.05, fixed_lengts=self.valid_lengths)
        self.feeder.roll_tape()

        self.current_lengths[best_node][:] = best_len
        self.freq_shift_times[best_node] = _time.currentTime

        sound = self.handle_controlled_sine(sound)
        gain = (self.ft.nodes.value.ravel()[None, :sound.shape[1]]
                * self.ft.main_volume.value).astype(np.float32)
        sound *= gain
        sound = self.out_filter(sound)

        # print(np.amax(sound[:, 0:2]))
        outdata[:] = sound

        self.show_output_volue(outdata)
        self.tlog.append([t0, time.time()])
        self.tlog.pop(0)

    def out_filter(self, sound):
        for channel in range(self.node_n_channels):
            low = self.ft.get_node_filter_low(channel)
            high = self.ft.get_node_filter_high(channel)
            sound[:, channel], self.zi[:, channel] = signal.lfilter(
                *signal.butter(3, (low, high), 'bandpass', fs=self.fs),
                sound[:, channel],
                axis=0, zi=self.zi[:, channel])
        return sound

    def get_current_tones(self, now):
        for i in range(self.node_n_channels):
            shift_mode, mean_duration, value = self.ft.get_shift_rate(i)
            if shift_mode == 0:
                if now - self.freq_shift_times[i] < mean_duration/2:
                    continue
                prob = 1-np.exp(-self.bsize/(self.fs*mean_duration/2))
                if np.random.random() < prob:
                    self.current_lengths[i] = np.random.choice(
                        [length for length in self.valid_lengths[i]
                         if length != self.current_lengths[i]],
                        size=1)
                    self.ft.nodes.ravel()[i].get_property(
                        'params')[0, 3].flash_color(ft_colors.blue)
                    self.freq_shift_times[i] = now
            else:
                idx = round(to_range(value, 0, len(self.valid_tones[i])-1))
                self.current_lengths[i] = self.valid_lengths[i][idx:idx+1]
        return self.current_lengths

    def handle_controlled_sine(self, sound):
        for i in range(self.node_n_channels):
            sine_mode, sig_shape, sine_tone = self.ft.get_contolled_tone(i)
            if sine_mode == 0:
                continue
            else:
                end_time = (self.bsize/self.fs) * 2*np.pi*sine_tone
                t = np.linspace(0, end_time, self.bsize) + self.sine_times[i]
                generator = [np.sin, signal.sawtooth,
                             signal.square][sig_shape]
                signal_gain = [1, 2/3, 1 / 2][sig_shape]
                sine = generator(t)*signal_gain
                self.sine_times[i] += end_time * (self.bsize+1)/self.bsize
                sound[:, i] = sine
        return sound

    def cb_mic(self, indata, _frames, _time, _status):
        # print(indata.shape)
        self.mic_slot.set(np.random.random(indata.shape))

    def show_input_volue(self, mean_squared):
        inputs = self.moniotors[:2, :3].ravel()
        ms = mean_squared.ravel()
        inputs[:ms.size].show_value(ms)

    def show_output_volue(self, outdata):
        outputs = self.moniotors[2:, :3].ravel()
        outputs[:outdata.shape[1]].show_value(np.amax(outdata, axis=0))

    def run(self):
        with self.ft, self.sd_stream_nodes:
            self.ft.do_task_delay(1000, self.repeat_foo)
            input('enter to quit')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    bro = Bro()
    bro.run()
