from swd_model import *
from read_TUSZ import *
import nitime.algorithms as tsa
import pandas as pd
from sklearn.model_selection import train_test_split

seed(162)
tf.random.set_seed(162)


def training(Xtrain, labels, test_set, test_labels, leave_n_out, train_num, target_path, in_size, out_size,
             conf_mat_write=True):
    """
    The function to conduct individual trainings
    :param Xtrain: Input data that goes into the model during training
    :param labels: Original input labels (one-hot)
    :param test_set: Input data that goes into the model during testing
    :param test_labels: Original test labels (one-hot)
    :param leave_n_out: the number of patients that is left to testing
    :param train_num: it counts the number from 0 to the number of folds in LOOCV conf.
    :param target_path: the path that the model, checkpoints, and history will be saved, should end with /
    :param in_size: the length of the input trial on the time/frequency axis
    :param out_size: output dimension, e.g. 2 for one-hot with 2 classes
    :param conf_mat_write: a boolean specifies whether to write confusion matrix
    :return:
    """
    # prepare the data
    x_train, x_val, y_train, y_val = train_test_split(Xtrain, labels, stratify=labels, test_size=0.3, random_state=1)

    Xtrain_II = x_train[:, 0, :]  # Channel-1: F7-T3 (train)
    Xtrain_V5 = x_train[:, 1, :]  # Channel-2: F8-T4 (train)

    xval_II = x_val[:, 0, :]  # Channel-1: F7-T3 (validation)
    xval_V5 = x_val[:, 1, :]  # Channel-2: F8-T4 (validation)

    # model definition
    model = define_model(in_size, out_size)

    # define callbacks
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    checkpoint_filepath = "leave_" + str(leave_n_out) + "_out_training_number_" + str(train_num)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=target_path + "checkpoint_" + checkpoint_filepath,
        save_weights_only=False,
        monitor='val_f1_score', mode='max',
        save_best_only=True)
    stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100, verbose=1, mode='max',
                                               baseline=None, restore_best_weights=True)
    where_am_I = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.1, patience=75, verbose=1,
                                                      mode='max', min_delta=0.001, cooldown=0, min_lr=0)
    # model training
    history = model.fit(x=[Xtrain_II, Xtrain_V5], y=y_train, epochs=500, batch_size=80, verbose=1,
                        validation_data=([xval_II, xval_V5], y_val),
                        callbacks=[model_checkpoint_callback, stop_me, where_am_I])  # TODO

    # save model and history info
    pd.DataFrame.from_dict(history.history).to_csv(target_path + str(checkpoint_filepath) + ".csv", index=False)
    model.save(target_path +  "checkpoint_" +checkpoint_filepath)
    print(
        "Complete: training num: " + str(train_num) + " for leave " + str(leave_n_out) + " out cross validation conf.")
    # print test results
    model.evaluate(x=[test_set[:, 0, :], test_set[:, 1, :]], y=test_labels, verbose=1)
    if conf_mat_write:
        predictions = model.predict(x=[test_set[:, 0, :], test_set[:, 1, :]])  # get the predictions
        predictions_decoded = (predictions[:, 0] < predictions[:, 1]) * 1  # decode one hot predictions to a vector
        test_labels_decoded = np.where(test_labels == 1)[1]  # get the original test labels as a vector
        thE_matrix = tf.math.confusion_matrix(labels=test_labels_decoded, predictions=predictions_decoded)
        print('Confusion Matrix: ')
        print(thE_matrix)


def time_leave_n_out_cross_validation(leave_n_out, npz_dataset_location, target_path, sample_dim=(1, 2, 2500)):
    """
    The function operates to realize leave n out cross validation for a dataset, time is the input of the proposed model
    :param leave_n_out: for LOOCV this value should be 1
    :param npz_dataset_location: the path/name of the windowed ready to train dataset
    :param target_path:target folder where the model, checkpoint and history will be saved
    :param sample_dim: the shape of one trial in the trial
    :return:
    """
    dataset = np.load(npz_dataset_location, allow_pickle=True)['patients'].item()  # get the data
    patients = list(dataset.keys())  # get the patient names
    num_of_training = int(len(patients) / leave_n_out)  # the num of training for leave n out cross validation conf.
    for train in range(0, num_of_training):
        test_patients = patients[train * leave_n_out:(train + 1) * leave_n_out]
        train_patients = list(set(patients).difference(set(test_patients)))
        train_patients.sort()

        train_dataset = np.zeros(sample_dim, float)  # dummy variable, initialization
        train_labels = np.zeros(1, int)  # dummy variable, initialization
        test_dataset = np.zeros(sample_dim, float)  # dummy variable, initialization
        test_labels = np.zeros(1, int)  # dummy variable, initialization

        for pats in train_patients:
            train_dataset = np.concatenate((train_dataset, dataset[pats]['data']))
            train_labels = np.concatenate((train_labels, dataset[pats]['labels']))
        for pats in test_patients:
            test_dataset = np.concatenate((test_dataset, dataset[pats]['data']))
            test_labels = np.concatenate((test_labels, dataset[pats]['labels']))

        # gets rid of the dummy variable
        train_dataset = train_dataset[1:, :, :]
        train_labels = train_labels[1:]
        test_dataset = test_dataset[1:, :, :]
        test_labels = test_labels[1:]

        # convert labels to one-hot
        one_hot_train_labels = np.zeros((train_labels.size, int(train_labels.max()) + 1))
        one_hot_train_labels[np.arange(train_labels.size), train_labels] = 1
        one_hot_test_labels = np.zeros((test_labels.size, int(test_labels.max()) + 1))
        one_hot_test_labels[np.arange(test_labels.size), test_labels] = 1

        training(train_dataset, one_hot_train_labels, test_dataset, one_hot_test_labels, leave_n_out, train,
                 target_path,
                 in_size=sample_dim[2], out_size=2, conf_mat_write=True)


def multitaperpsd_leave_n_out_cross_validation(leave_n_out, npz_dataset_location, target_path, sample_dim=(1, 2, 1251)):
    """
    The function operates to realize leave n out cross validation for a dataset,
    power spectral density is the input of the proposed model
    :param target_path:target folder where the model, checkpoint and history will be saved
    :param leave_n_out: for LOOCV this value should be 1
    :param npz_dataset_location: the path/name of the windowed ready to train dataset
    :param sample_dim: the shape of one trial in the trial
    :return:
    """
    dataset = np.load(npz_dataset_location, allow_pickle=True)['patients'].item()  # get the data
    patients = list(dataset.keys())  # get the patient names
    num_of_training = int(len(patients) / leave_n_out)  # the num of training for leave n out cross validation conf.
    for train in range(0, num_of_training):
        test_patients = patients[train * leave_n_out:(train + 1) * leave_n_out]
        train_patients = list(set(patients).difference(set(test_patients)))
        train_patients.sort()

        train_dataset = np.zeros(sample_dim, float)  # dummy variable, initialization
        train_labels = np.zeros(1, int)  # dummy variable, initialization
        test_dataset = np.zeros(sample_dim, float)  # dummy variable, initialization
        test_labels = np.zeros(1, int)  # dummy variable, initialization

        for pats in train_patients:
            # applies multitaper and gets power spectral density as psd_mt
            # original data sampling frequency=250 Hz
            f, psd_mt, nu = tsa.multi_taper_psd(dataset[pats]['data'], adaptive=False, jackknife=False, Fs=250, NW=6)
            # replaces zeros with epsilon to enable log operation
            psd_mt = np.where(psd_mt > 0.0000000001, psd_mt, 0.0000000001)
            train_dataset = np.concatenate((train_dataset, np.log10(psd_mt)))  # gets the log10
            train_labels = np.concatenate((train_labels, dataset[pats]['labels']))
        for pats in test_patients:
            # applies multitaper and gets power spectral density as psd_mt
            # original data sampling frequency=250 Hz
            f, psd_mt, nu = tsa.multi_taper_psd(dataset[pats]['data'], adaptive=False, jackknife=False, Fs=250, NW=6)
            # replaces zeros with epsilon to enable log operation
            psd_mt = np.where(psd_mt > 0.0000000001, psd_mt, 0.0000000001)
            test_dataset = np.concatenate((test_dataset, np.log10(psd_mt)))  # gets the log10
            test_labels = np.concatenate((test_labels, dataset[pats]['labels']))

        # gets rid of the dummy variable
        train_dataset = train_dataset[1:, :, :]
        train_labels = train_labels[1:]
        test_dataset = test_dataset[1:, :, :]
        test_labels = test_labels[1:]

        # convert labels to one-hot
        one_hot_train_labels = np.zeros((train_labels.size, int(train_labels.max()) + 1))
        one_hot_train_labels[np.arange(train_labels.size), train_labels] = 1
        one_hot_test_labels = np.zeros((test_labels.size, int(test_labels.max()) + 1))
        one_hot_test_labels[np.arange(test_labels.size), test_labels] = 1

        training(train_dataset, one_hot_train_labels, test_dataset, one_hot_test_labels, leave_n_out, train,
                 target_path,
                 in_size=sample_dim[2], out_size=2, conf_mat_write=True)


def assume(original_labels, predictions):
    assumed = predictions[:]
    differences = np.where(original_labels != predictions)[0]
    differences.sort()
    for difference in differences:
        if original_labels[difference - 1] != original_labels[difference + 1]:
            # means there is transition from 0 to 1 or 1 to 0, thus assumptions are applicable
            assumed[difference] = original_labels[difference]
        elif (original_labels[difference - 1] and original_labels[difference + 1]):
            assumed[difference] = 1
    return assumed


def apply_assumptions(models_folder, dataset_path, sample_dim=(1, 2, 1251), is_time=False):
    dataset = np.load(dataset_path, allow_pickle=True)['patients'].item()
    patients = list(dataset.keys())
    num_of_set = len(patients)
    num_of_training = int(num_of_set)
    for train_num in range(0, num_of_training):
        test_patients = patients[train_num:(train_num + 1)]
        test_dataset = np.zeros(sample_dim, float)  # dummy variable, initialization
        test_labels = np.zeros(1, int)  # dummy variable, initialization
        data = dataset[test_patients[0]]['data']
        if not is_time:
            data = data2input(data)
        test_dataset = np.concatenate((test_dataset, data))
        test_labels = np.concatenate((test_labels, dataset[test_patients[0]]['labels']))

        # gets rid of the dummy variable
        test_dataset = test_dataset[1:, :, :]
        test_labels = test_labels[1:]
        # convert labels to one hot
        one_hot_test_labels = np.zeros((test_labels.size, int(test_labels.max()) + 1))
        one_hot_test_labels[np.arange(test_labels.size), test_labels] = 1

        #  eval
        model_path = models_folder + "/checkpoint_leave_" + str(1) + "_out_training_number_" + str(train_num) + "/"
        model = tf.keras.models.load_model(model_path, custom_objects={"F1Score": tfa.metrics.F1Score}, compile=True)
        print(model_path)
        # model.evaluate(x=[test_dataset[:, 0, :], test_dataset[:, 1, :]], y=one_hot_test_labels, verbose=1,sample_weight=None)
        # confusion matrix
        predictions = model.predict(x=[test_dataset[:, 0, :], test_dataset[:, 1, :]])
        predictions_decoded = (predictions[:, 0] < predictions[:, 1]) * 1
        test_labels_decoded = np.where(one_hot_test_labels == 1)[1]
        assumed = assume(original_labels=test_labels_decoded, predictions=predictions_decoded)
        thE_matrix2 = tf.math.confusion_matrix(labels=test_labels_decoded, predictions=assumed)
        print('Confusion Matrix for: ' + test_patients[0])
        print(thE_matrix2)


def data2input(data):
    f, psd_mt, nu = tsa.multi_taper_psd(data, adaptive=False, jackknife=False, Fs=250, NW=6)
    psd_mt = np.where(psd_mt > 0.0000000001, psd_mt, 0.0000000001)
    return np.log10(psd_mt)


def calculate_metrics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TN + FP + FN + TP)
    sensitivity = TP / (FN + TP)
    specifity = TN / (FP + TN)
    FD = FP * 360 / (TN + FP + FN + TP)
    print('accuracy: ' + str(accuracy))
    print('sensitivity: ' + str(sensitivity))
    print('specifity: ' + str(specifity))
    print('FD: ' + str(FD))


def adjust_absz_patients(absz_dir, target_name, config, Fs=250):
    """
    :param file_dir: # the dir where absz .npz files exists, ex: file_dir = "../absz/"
    :param target_name: # the dir where all patients will be stored ex: target_name = "absz_patients"
    :return:
    """
    current_dir = os.getcwd()
    os.chdir(absz_dir)
    # the following three can be defined inside the loop to generalize for all sampling frequency
    sampling_frequency = Fs
    windowing_time = config['window_width']
    sample_per_label = sampling_frequency * windowing_time
    overlapping = config['overlap']
    shift = windowing_time * overlapping
    all_patient_data = {}
    for patients in sorted(os.listdir()):
        patient_data = np.load(patients, allow_pickle=True)['patient'].item()
        data = []
        labels = []
        for sessions in patient_data.keys():
            for records in patient_data[sessions].keys():
                record = patient_data[sessions][records]
                num_of_labels = len(record['label'])
                F7_T3 = record['data'][10, :] - record['data'][12, :]
                F8_T4 = record['data'][11, :] - record['data'][13, :]
                label = (record['label'][:] == 5) + 0
                for row in range(0, num_of_labels):
                    eeg_signal_beginning = int(row * (windowing_time - shift) * sampling_frequency)
                    eeg_signal_end = int(eeg_signal_beginning + sample_per_label)
                    data.append(
                        [F7_T3[eeg_signal_beginning:eeg_signal_end], F8_T4[eeg_signal_beginning:eeg_signal_end]])
                    labels.append(label[row])
        all_patient_data[patients] = {}
        all_patient_data[patients]['data'] = np.array(data)
        all_patient_data[patients]['labels'] = np.array(labels)
    np.savez(target_name + '.npz', patients=all_patient_data)
    os.chdir(current_dir)
