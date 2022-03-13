# SWD-Detect-with-SVM-and-tree

This directory contains the implementations of feature extraction and SWD detection methods proposed in the cited works below:
* [An automated, machine learning-based detection algorithm for spike-wave discharges (SWDs) in a mouse model of absence epilepsy](https://www.biorxiv.org/content/10.1101/309146v3)
* [Discrete wavelet transform for analysis of spike-wave discharges in rats](https://ieeexplore.ieee.org/document/4650257)

## Repo Description 
A 12 feature SVM implementation of SWD Detection algorithm implemented in MATLAB R2021b. 

## Data 
The directories in [MAT and absz](https://drive.google.com/drive/folders/1vaUe2F92PrSW9aJ2wb3HlJAm2VHjDsR9?usp=sharing) should be downloaded to this directory to parse the data and create feature matrices. 

## Scripts

* First, run the feature extraction scripts [create_human_features.m](https://github.com/Berken-demirel/SWD_Detect/blob/master/Comparison%20to%20Other%20Works/CWT/create_human_features.m) and [create_rat_features.m](https://github.com/Berken-demirel/SWD_Detect/blob/master/Comparison%20to%20Other%20Works/CWT/create_rat_features.m) scripts for both CWT and DWT directories to generate the feature matrices. 

* [humans_run.m](https://github.com/Berken-demirel/SWD_Detect/blob/master/Comparison%20to%20Other%20Works/CWT/humans_run.m) and [rats_run.m](https://github.com/Berken-demirel/SWD_Detect/blob/master/Comparison%20to%20Other%20Works/CWT/rats_run.m) scripts are run to generate, output and save the classification results in the form of a table, saved in .mat files.
