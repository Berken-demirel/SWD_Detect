import numpy as np
from glob import glob
import os

# Creates training and test data leaving the animal out based on the animal_idi along with the number of samples epilepsy is observed in a segment. 

def create_training_data_PSD(animal_id: int):

    os.chdir("ALL_RATS_RAW_DATA//")

    CH1_files = glob("*CH1.npy")
    CH1_files = np.array(
        list(filter(lambda item: item[0:3] == "PSD", CH1_files)))
    animals = list(map(lambda item: int(item.split("_")[1][6:]), CH1_files))
    CH2_files = glob("*CH2.npy")
    CH2_files = np.array(
        list(filter(lambda item: item[0:3] == "PSD", CH2_files)))
    label_files = np.array(
        list(filter(lambda item: item.endswith(".npy"), glob("*labels*"))))

    test_idx = animals.index(animal_id)
    train_indices = np.array(list((set(range(len(animals)))-{test_idx})))
    Train_CH1 = CH1_files[train_indices]
    Test_CH1 = CH1_files[test_idx]
    Train_CH2 = CH2_files[train_indices]
    Test_CH2 = CH2_files[test_idx]
    Train_Labels = label_files[train_indices]
    Test_Labels = label_files[test_idx]

    Train_CH1 = [np.load(item, allow_pickle=True).item()[
        "PSD_CH1"] for item in Train_CH1]
    Train_CH1 = np.vstack(Train_CH1)
    Test_CH1 = np.load(Test_CH1, allow_pickle=True).item()["PSD_CH1"]

    Train_CH2 = [np.load(item, allow_pickle=True).item()[
        "PSD_CH2"] for item in Train_CH2]
    Train_CH2 = np.vstack(Train_CH2)
    Test_CH2 = np.load(Test_CH2, allow_pickle=True).item()["PSD_CH2"]

    Train_Labels = np.hstack([np.load(item, allow_pickle=True).item()[
                             "Final_labels"] for item in Train_Labels])
    Test_Labels = np.load(label_files[test_idx], allow_pickle=True).item()[
        "Final_labels"]

    Train_Labels = Train_Labels[0, :]
    Test_Times = Test_Labels[1, :]
    Test_labels = Test_Labels[0, :]

    os.chdir("..")

    return Train_CH1, Test_CH1, Train_CH2, Test_CH2, Train_Labels, Test_labels, Test_Times


def create_training_data_time(animal_id: int):

    os.chdir("ALL_RATS_RAW_DATA//")

    CH1_files = glob("*CH1.npy")
    CH1_files = np.array(
        list(filter(lambda item: not item[0:3] == "PSD", CH1_files)))
    animals = list(map(lambda item: int(item.split("_")[1][6:-7]), CH1_files))
    CH2_files = glob("*CH2.npy")
    CH2_files = np.array(
        list(filter(lambda item: not item[0:3] == "PSD", CH2_files)))
    label_files = np.array(
        list(filter(lambda item: item.endswith(".npy"), glob("*labels*"))))

    test_idx = animals.index(animal_id)
    train_indices = np.array(list((set(range(len(animals)))-{test_idx})))
    Train_CH1 = CH1_files[train_indices]
    Test_CH1 = CH1_files[test_idx]
    Train_CH2 = CH2_files[train_indices]
    Test_CH2 = CH2_files[test_idx]
    Train_Labels = label_files[train_indices]
    Test_Labels = label_files[test_idx]

    Train_CH1 = [np.load(item, allow_pickle=True).item()[
        "Voltage_CH1"] for item in Train_CH1]
    Train_CH1 = np.vstack(Train_CH1)
    Test_CH1 = np.load(Test_CH1, allow_pickle=True).item()["Voltage_CH1"]

    Train_CH2 = [np.load(item, allow_pickle=True).item()[
        "Voltage_CH2"] for item in Train_CH2]
    Train_CH2 = np.vstack(Train_CH2)
    Test_CH2 = np.load(Test_CH2, allow_pickle=True).item()["Voltage_CH2"]

    Train_Labels = np.hstack([np.load(item, allow_pickle=True).item()[
                             "Final_labels"] for item in Train_Labels])
    Test_Labels = np.load(label_files[test_idx], allow_pickle=True).item()[
        "Final_labels"]

    Train_Labels = Train_Labels[0, :]
    Test_Times = Test_Labels[1, :]
    Test_labels = Test_Labels[0, :]

    os.chdir("..")

    return Train_CH1, Test_CH1, Train_CH2, Test_CH2, Train_Labels, Test_labels, Test_Times
