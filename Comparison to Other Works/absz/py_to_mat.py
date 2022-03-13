# %%
import numpy as np 
from scipy.io import savemat
from glob import glob

patient_data = np.load("absz_all.npz", allow_pickle=True)['patients'].item()
files = patient_data.keys()

# %% 

for patient in files:
    ID = patient[8:13]
    data_and_label = patient_data[patient]
    data = data_and_label["data"]
    data = data[:,0,:]
    labels = data_and_label["labels"]
    Voltage_CH1 = {}
    Voltage_CH1["data"] = data
    Voltage_CH1["labels"] = labels
    savemat("HumanData_"+str(ID)+".mat",Voltage_CH1)

# %%
