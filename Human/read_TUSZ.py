import pyedflib
import numpy as np
import os
import re


def read_tse(file_name, subfolder):  # input without extensions
    symbols = {'(null)': 0, 'spsw': 1, 'gped': 2, 'pled': 3, 'eyem': 4, 'artf': 5, 'bckg': 6, 'seiz': 7, 'fnsz': 8,
               'gnsz': 9, 'spsz': 10, 'cpsz': 11, 'absz': 12, 'tnsz': 13, 'cnsz': 14, 'tcsz': 15, 'atsz': 16,
               'mysz': 17, 'nesz': 18, 'intr': 19, 'slow': 20, 'eyem': 21, 'chew': 22, 'shiv': 23, 'musc': 24,
               'elpp': 25, 'elst': 26, 'calb': 27}

    fp = open(file_name + '.tse', 'r')
    lines = fp.readlines()
    useful_data = []
    for line in lines:
        if line[0].isdigit():
            numeric_line = line.split()
            if numeric_line[2] != 'bckg':
                record = {'start_time': float(numeric_line[0]),
                          'end_time': float(numeric_line[1]),
                          'seiz_type': numeric_line[2],
                          }
                record['idx_seiz_type'] = symbols[record['seiz_type']]
                channels = channel_detect(file_name, record['idx_seiz_type'], record['start_time'], record['end_time'],
                                          subfolder)
                record['channels'] = channels
                useful_data.append(record)
                # probability = numeric_line[3]
    fp.close()
    return useful_data


def channel_detect(file_dir, seiz_idx, tse_start_time, tse_end_time, subfolder):
    # following is the desired channels
    channels_ar_a = {0: 'FP1 - F7',
                     1: 'F7 - T3',
                     2: 'T3 - T5',
                     3: 'T5 - O1',
                     4: 'FP2 - F8',
                     5: 'F8 - T4',
                     6: 'T4 - T6',
                     7: 'T6 - O2',
                     8: 'T3 - C3',
                     9: 'C3 - CZ',
                     10: 'CZ - C4',
                     11: 'C4 - T4',
                     12: 'FP1 - F3',
                     13: 'F3 - C3',
                     14: 'C3 - P3',
                     15: 'P3 - O1',
                     16: 'FP2 - F4',
                     17: 'F4 - C4',
                     18: 'C4 - P4',
                     19: 'P4 - O2'}

    if subfolder == '01_tcp_ar' or subfolder == '02_tcp_le':
        desired = False
        # channels = channels_ar_le
    elif subfolder == '03_tcp_ar_a':
        desired = True
        # channels = channels_ar_a
    else:
        print('AN ERROR OCCURRED, GIVEN SET DOESNT EXIST')
        return None

    f = open(file_dir + '.lbl', 'r')
    lines = f.readlines()
    detected_channels_string = []
    detected_channels_onehot = np.zeros((len(channels_ar_a)))  # TODO correction needed to fix number
    for line in lines:
        if line[0:5] == 'label':
            label_line = re.split('\[|\]', line)
            labels = label_line[1].split(',')
            labels = [int(float(numeric_string)) for numeric_string in labels]  # gets the one hot labels for seizures
            if labels[seiz_idx]:
                info = label_line[0].split(',')
                channel_num = int(info[-2])
                end = float(info[-3])
                start = float(info[-4])
                is_A1_A2_related = ((not desired) and (channel_num == 8 or channel_num == 13))
                if start >= tse_start_time and end <= tse_end_time and (not is_A1_A2_related):
                    # only channels in this given interval, preventing adding channels detected later for the same seizure
                    if not desired:
                        converted_channel_num = channel_num
                        if channel_num > 8:
                            converted_channel_num -= 1
                        if channel_num > 13:
                            converted_channel_num -= 1
                        channel_num = converted_channel_num

                    detected_channels_onehot[channel_num] = 1
                    channel_name = channels_ar_a[channel_num]
                    detected_channels_string.append(channel_name)
    f.close()
    return {'names': detected_channels_string, 'onehot': detected_channels_onehot}


def reformat_raw_edfs(edf_file_name, subfolder):  # TODO comment it
    essential_signals_ar = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                            'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                            'EEG PZ-REF']

    essential_signals_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE',
                            'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE',
                            'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE']
    if subfolder == '01_tcp_ar' or subfolder == '03_tcp_ar_a':
        essential_signals = essential_signals_ar
    elif subfolder == '02_tcp_le':
        essential_signals = essential_signals_le
    else:
        print('AN ERROR OCCURRED, GIVEN SET DOESNT EXIST')
        return None

    edf_file = read_single_edf(edf_file_name + '.edf')
    sampling_frequency = edf_file['sample_frequency']
    signal_duration = edf_file['duration']
    signal_data = edf_file['sigbufs']
    signal_labels = edf_file['signal_labels']
    data_length = np.shape(signal_data)[1]
    data_features_channels = len(essential_signals)
    output = np.zeros((data_features_channels, data_length))
    for i in range(len(essential_signals)):
        current_signal = essential_signals[i]
        output[i] = signal_data[signal_labels.index(current_signal)]
    return output, sampling_frequency, signal_duration


def read_single_edf(name_with_dir):
    f = pyedflib.EdfReader(name_with_dir)
    frequency = f.getSampleFrequency(0)  # desired sampling frequency
    frequencies = f.getSampleFrequencies()  # all sampling frequencies in signal set
    n = np.count_nonzero(frequencies == frequency)  # calculates the total number of useful signals
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))

    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    duration = f.file_duration
    output = {'f': f, 'n': n, 'signal_labels': signal_labels, 'sigbufs': sigbufs, 'duration': duration,
              'sample_frequency': frequency}
    f.close()
    return output


def create_output_labels(info_dictionary, file_name, window_width, sensitivity, overlap):
    label_dict = {'fnsz': 1, 'gnsz': 2, 'spsz': 3, 'cpsz': 4, 'absz': 5, 'tnsz': 6, 'cnsz': 7, 'tcsz': 8, 'atsz': 9,
                  'mysz': 10, 'nesz': 11}
    f = pyedflib.EdfReader(file_name + '.edf')
    signal_duration = f.file_duration  # in seconds
    f.close()
    minimum_awared_duration = sensitivity * window_width

    overlap_width = overlap * window_width
    frame_num = int((signal_duration - overlap_width) / (window_width - overlap_width))
    labels = np.zeros(frame_num)
    for detections in info_dictionary:
        start_time = detections['start_time']
        end_time = detections['end_time']
        seiz_label = label_dict[detections['seiz_type']]

        end_time_residual = end_time % (window_width - overlap_width)
        start_time_residual = (start_time - overlap_width) % (window_width - overlap_width)

        detection_start_frame = int((start_time - overlap_width) / (window_width - overlap_width))
        if (window_width - overlap_width) - start_time_residual < minimum_awared_duration:
            detection_start_frame += 1

        detection_end_frame = int(end_time / (window_width - overlap_width))
        if end_time_residual < minimum_awared_duration:
            detection_end_frame -= 1

        labels_can_present = np.array([0, seiz_label])
        if np.size(np.setdiff1d(labels[detection_start_frame:detection_end_frame + 1], labels_can_present)) != 0:
            print('An error occured, the different seizures are overlapped..')
            print(detections)
            print(labels[detection_start_frame:detection_end_frame + 1])
            return
        labels[detection_start_frame:detection_end_frame + 1] = seiz_label
    return labels


def gather_sample_data(sample_name, config, subfolder):
    sample_dict = read_tse(sample_name, subfolder)
    [sample_eeg, sample_freq, signal_duration] = reformat_raw_edfs(sample_name, subfolder)
    label = create_output_labels(sample_dict, sample_name,
                                 config['window_width'], config['sensitivity'], config['overlap'])
    return {'info': sample_dict, 'data': sample_eeg, 'label': label, 'freq': sample_freq, 'duration': signal_duration}


def gather_session_data(session_folder, config, subfolder):
    session_dict = {}
    os.chdir(session_folder)
    for file in sorted(os.listdir()):
        if file.endswith('.edf'):
            file_name = file[:-4]
            processed_sample = gather_sample_data(file_name, config, subfolder)
            sample_key = file_name[-4:]
            session_dict[sample_key] = processed_sample
    os.chdir('../')
    return session_dict


def gather_patient_data(patient_folder,
                        config, save,
                        target_path,
                        ):  # input dir and target path shouldn't end with / and full name of these folders are needed
    os.chdir(patient_folder)
    patient_dict = {}
    name = re.sub(r"\s+", "", patient_folder).split('/')
    file_name = str(name[-2]) + "_" + str(name[-1])
    subfolder = str(name[-3])

    for sessions in sorted(os.listdir()):
        print('current session: ' + str(sessions))
        session_dict = gather_session_data(sessions, config, subfolder)
        session_key = sessions[:4]
        patient_dict[session_key] = session_dict

    if save:
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        os.chdir(target_path)
        np.savez_compressed(file_name + ".npz", patient=patient_dict)
        # data = np.load(path, allow_pickle=True)['patient'].item() to reach data
        return

    return patient_dict


def prepare_database(edf_location, target_folder, config, only_absz=True):
    # edf_location=location of the edf folder
    # target_path=
    absz_pats = ['/dev/02_tcp_le/006/00000675',
                 '/train/02_tcp_le/011/00001113',
                 '/train/02_tcp_le/014/00001413',
                 '/train/02_tcp_le/017/00001795',
                 '/dev/02_tcp_le/019/00001984',
                 '/train/02_tcp_le/024/00002448',
                 '/train/02_tcp_le/026/00002657',
                 '/train/02_tcp_le/030/00003053',
                 '/dev/02_tcp_le/032/00003281',
                 '/dev/02_tcp_le/033/00003306',
                 '/dev/02_tcp_le/036/00003635',
                 '/train/01_tcp_ar/086/00008608']
    current_dir = os.getcwd()
    target_folder = current_dir+"/"+target_folder
    edf_location = edf_location+"edf"
    if only_absz:
        for patients in absz_pats:
            print('Processing... '+patients)
            os.chdir(edf_location+patients)
            patient_folder = os.getcwd()
            gather_patient_data(patient_folder=patient_folder,
                                config=config, save=True,
                                target_path=target_folder)
            os.chdir(current_dir)

    else:
        os.chdir(edf_location)
        for sets in sorted(os.listdir()):
            os.chdir(sets)
            for set_types in sorted(os.listdir()):
                os.chdir(set_types)
                for hospitals in sorted(os.listdir()):
                    os.chdir(hospitals)
                    for patients in sorted(os.listdir()):
                        print(
                            'Processing... patient: ' + patients + " hospital: " + hospitals)
                        os.chdir(patients)
                        patient_folder = os.getcwd()
                        gather_patient_data(patient_folder=patient_folder,
                                            config=config, save=True,
                                            target_path=target_folder)
                        os.chdir(patient_folder)
                        os.chdir('../')
                    os.chdir('../')
                os.chdir('../')
            os.chdir('../')
