from scipy.io.wavfile import write as writewav
import numpy as np
from pathlib import Path

datafolder = Path(__file__).parent
wavfolder = datafolder.joinpath('wavfiles')

npz_files = sorted([i for i in datafolder.iterdir() if i.suffix == '.npz'])
file = npz_files[1]
for file in npz_files:
    with np.load(file) as data:
        hello = data['arr_0'].astype(np.float32)

    writewav(str(wavfolder.joinpath(file.stem + '.wav')), 48000, hello[:, 2])
# for file in wavfolder.
