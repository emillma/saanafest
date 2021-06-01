from scipy import signal
import sounddevice as sd
import numpy as np
from scipy.fftpack import rfft, irfft
from feedback import Feeder
from time import time as now
from matplotlib import pyplot as plt


fs = 48000
BSIZE = 2048
CHANNELS_OUT = 2
CHANNELS_IN = 3
feeder = Feeder(CHANNELS_OUT, BSIZE, fs)

# g_note = 196/2
# filter_freqs = np.array([g_note-10, g_note+10])
filt_b, filt_a = signal.butter(3, 1100, 'lowpass', fs=fs)
filt_bhigh, filt_ahigh = signal.butter(3, 50, 'highpass', fs=fs)
zi = np.stack(CHANNELS_IN*[signal.lfilter_zi(filt_b, filt_a)], -1)
zihigh = np.stack(CHANNELS_OUT*[signal.lfilter_zi(filt_bhigh, filt_ahigh)], -1)

counter = [0]
safety = [1]


def callback(indata, outdata, frames, time, status) -> None:
    indata, zi[:] = signal.lfilter(
        filt_b, filt_a,
        indata,
        axis=0, zi=zi)

    t0 = now()
    contacts_in = indata[:, 0:2]
    mic_in = indata[:, 2:3]

    freqs = np.abs(rfft(contacts_in, axis=0))
    contact_amps = np.amax(
        freqs*np.arange(freqs.shape[0])[:, None], axis=0)/(fs/440)
    contact_gains = np.maximum(0, (1-(contact_amps/5)))[None, :]**2 * 50
    if np.amax(mic_in) < 0.01:
        alpha = 0.01
    else:
        alpha = 0.02

    sound = feeder.step(contacts_in*contact_gains +
                        mic_in*20, alpha=alpha) * 5

    sound, zihigh[:] = signal.lfilter(
        filt_bhigh, filt_ahigh,
        sound,
        axis=0, zi=zihigh)

    phases = np.array([[0, np.pi]])
    times = np.stack(
        sound.shape[1]*[np.linspace(now(), now()+BSIZE/fs, BSIZE)], axis=1)

    phasegains = np.cos(phases + times * 2*np.pi * 0.1)/2 + 0.5
    phaseamp = 0.2
    phasegains = 1 - phaseamp + phaseamp*phasegains

    sound *= phasegains
    sound *= np.array([[1, 1]])

    if np.amax(np.abs(sound)) > 1:
        print('to much!')
        safety[0] = 0
        sound *= 0

    outdata[:] = sound * safety[0]

    print(
        f'{np.amax(contact_amps[0]):.3f}',
        f'{np.amax(contact_amps[1]):.3f}',
        f'{np.amax(mic_in):.3f}',
        f'{np.amax(outdata):.3f}',
        f'{contact_gains[0,0]:.3f}',
        f'{contact_gains[0,1]:.3f}',
    )


# audiobox_in = 'Microphone (AudioBox 44 VSL ), Windows WDM-KS'
# audiobox_out = 'Speakers (AudioBox 44 VSL ), Windows WDM-KS'

audiobox_in = 'Microphone (AudioBox 44 VSL ), MME'
audiobox_out = 'Speakers (AudioBox 44 VSL ), MME'
yeti_in = 'Microphone (Yeti Stereo Microph, MME'
pc_out = 'Speakers (Realtek High Definiti, MME'
sonywh_out = 'Headphones (WH-1000XM2 Stereo), MME'

with sd.Stream(samplerate=fs,
               blocksize=BSIZE,
               device=(audiobox_in, audiobox_out),
               channels=(CHANNELS_IN, CHANNELS_OUT),
               dtype=np.float32,
               latency='low',
               callback=callback,

               ):
    input('enter to quit')
plt.show()
