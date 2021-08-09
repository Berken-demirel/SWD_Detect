import glob
import os
from scipy.io import loadmat
import numpy as np


def MATLAB2Python(path: str):
    os.chdir(path)
    files = glob.glob("*.mat")
    print(files)
    for file in files:
        data = loadmat(file)
        np.save(file[:-4], data)
        print("saved")


MATLAB2Python("ALL_RATS_RAW_DATA")
