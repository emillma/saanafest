import numpy as np
from pathlib import Path

# src_folder = Path(__file__).parent
data_folder = Path(__file__).parents[1].joinpath('data')
for file in data_folder.iterdir():
    if file.suffix == '.txt':
        # print(file.stem)
        array = np.loadtxt(file)
        np.savez_compressed(data_folder.joinpath(file.stem+'.npz'), array)
