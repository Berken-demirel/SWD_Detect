# SWD-Detection-in-Humans
## Initial Setup
#### The model files are transferred to the Google Drive due to Git LFS restrictions.
#### One should download the folders in the Google Drive URL below and "replace" with the corresponding folders in the GitHub repository before running the scripts to get meaningful results. 

[Google Drive URL](https://drive.google.com/drive/folders/1U82sRliO3sm058cut5ywFu_Q3S3F8s9s?usp=sharing)

#### Furhtermore, one should download the edf folder of the TUH's "TUSZ Corpus" from the link below and this folder should be replaced with the "edf" folder in the GitHub repository. (Note that it is requited to register to TUH database because they would like to trace the name of the user institutions/organizations) We used v1.5.2 of their TUSZ release during training. We will be using only two montages of their multi channel EEG data as shown in the Figure:
![Figure](./img/skull.png | width=100)

[Temple University Hospital TUSZ Dataset URL](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/)

[Temple University Hospital All Datasets URL](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)

#### Before running the scripts, please check the [requirements.txt](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/requirements.txt).

## Running Scripts
#### If one completed initial setup, the [main.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/main.py) can be directly run to see the evaluation results in the paper. Whole workflow is embedded into [main.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/main.py) so one do not have to deal with individual functions/scripts. Time consuming steps are commented out so that one can just run the [main.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/main.py) to get results. However, if one is interested in initial steps that we followed, please feel free to uncomment the codes inside the  [main.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/main.py). For further details about processes please visit [main.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/main.py).


## NN Model Architecture
#### The model architecture is implemented in [swd_model.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/swd_model.py). One should note that tensorflow_addons library is needed to run the script without errors.

## Utilities Script
#### [swd_utils.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/swd_utils.py) contains most of the important functions such as estimating the Multitaper PSD, configuration function provides Leave N One Out Cross Validation, metric calculation etc. All functions are explained inside the script just after the definition. Feel free to investigate further.  

## Reading the TUSZ Corpus of TUH
#### Since the data got from [TUSZ Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tusz) is not ready to feed into our neural network directly, some file handling functions are coded in [read_TUSZ.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/read_TUSZ.py). The functioncan be generalized for further applications and not only limited to absz seizure inside the corpus. There are a lot of arguments that can modify the main purpose of the functions. Researchers are free to use our functions according to their needs by citing our paper. Please note that [pyedflib](https://pyedflib.readthedocs.io/en/latest/contents.html) is necessary to run this script.
