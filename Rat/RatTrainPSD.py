import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from keras.models import Sequential
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from scipy import signal
import pandas as pd
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, SeparableConv1D, Dropout, Flatten, Concatenate, Reshape, \
    Activation, BatchNormalization, SeparableConv1D, Add, Activation, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam, SGD
import numpy as np
import os
import sys
import joblib
from scipy.signal import butter, sosfiltfilt
import math
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import tensorflow_addons as tfa
from sklearn.preprocessing import normalize
from create_training_data import create_training_data_PSD
from sklearn.metrics import confusion_matrix
import h5py

def scale_between_0_1(arr):
    return normalize(arr, norm="max")


def assign_weigths(sums):
    if np.count_nonzero(sums) != 2:
        sums[np.where(sums == 0)[0]] = 1
    total = np.sum(sums)
    sums_div = 1 / (sums / total)
    weights_dict = {}
    for i in range(len(sums)):
        weights_dict.__setitem__(i, sums_div[i] / 2)
    return weights_dict


def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def inception_module_1(layer_in):
    conv4 = Conv1D(32, 4, padding='same', activation='relu',
                   kernel_initializer='GlorotNormal')(layer_in)
    conv16 = Conv1D(32, 16, padding='same', activation='relu',
                    kernel_initializer='GlorotNormal')(layer_in)
    layer_out = concatenate([conv4, conv16], axis=-1)
    x3 = BatchNormalization()(layer_out)
    return x3


def res_net_block1(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation='relu',
               padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x


def res_net_block_trans(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu',
                         padding='same')(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu',
                padding='same')(input_data)
    x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Add()([x3, input_trans])
    x = Activation('relu')(x4)
    return x


def Lead_II_way(lead_II):
    layer_out = Conv1D(32, 8, activation='relu',
                       kernel_initializer='GlorotNormal', padding='same')(lead_II)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    #Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Pool1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    return Pool1


def Lead_V5_way(lead_V5):
    layer_out = Conv1D(32, 8, activation='relu',
                       kernel_initializer='GlorotNormal', padding='same')(lead_V5)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    #Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Pool1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # Pool1 = signal.decimate(res2, 2)
    # flat = Flatten()(Pool1)
    return Pool1


def define_model(in_shape=(1251, 1,), out_shape=2):
    input_II = Input(shape=in_shape)
    input_V5 = Input(shape=in_shape)

    out_II = Lead_II_way(input_II)
    out_V5 = Lead_V5_way(input_V5)
    layer_out = concatenate([out_II, out_V5], axis=-1)
    sep1 = SeparableConv1D(128, 4, activation='relu',
                           kernel_initializer='GlorotNormal', padding='same')(layer_out)
    flat = Flatten()(sep1)
    Dense_1 = Dense(128, activation='relu')(flat)
    Dropout1 = Dropout(0.4)(Dense_1)
    out = Dense(out_shape, activation='softmax')(Dropout1)
    model = Model(inputs=[input_II, input_V5], outputs=out)
    # compile model
    opt = Adam(learning_rate=0.0003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[
        'Recall', "accuracy", tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')])
    return model

## Leave-One-Record-Out Cross Validation Loop

for animal in [15, 16, 17, 86, 88, 89, 90, 91, 92, 103, 104]:

    Experiment = "TestAnimal" + str(animal)

    Train_CH1, Test_CH1, Train_CH2, Test_CH2, Train_Labels, Test_Labels, Test_Times = create_training_data_PSD(
        animal)

    Train_Labels = np.reshape(Train_Labels, newshape=(Train_Labels.size,))
    Test_Labels = to_categorical(np.reshape(
        Test_Labels, newshape=(Test_Labels.size,)))

    # Max Normalization
    Train_CH1 = scale_between_0_1(Train_CH1)
    Train_CH2 = scale_between_0_1(Train_CH2)
    Test_CH1 = scale_between_0_1(Test_CH1)
    Test_CH2 = scale_between_0_1(Test_CH2)

    # Added depth layer for convolution
    w, h = Train_CH1.shape
    Train_CH1 = np.reshape(Train_CH1, newshape=(w, h, 1))
    w, h = Train_CH2.shape
    Train_CH2 = np.reshape(Train_CH2, newshape=(w, h, 1))
    w, h = Test_CH1.shape
    Test_CH1 = np.reshape(Test_CH1, newshape=(w, h, 1))
    w, h = Test_CH2.shape
    Test_CH2 = np.reshape(Test_CH2, newshape=(w, h, 1))

    model = define_model(in_shape=(h, 1,))

    # Train Validation Split
    CHANNELS = np.concatenate((Train_CH1, Train_CH2), axis=1)
    CH_train, CH_val, CH_trainlabel, CH_vallabel = train_test_split(
        CHANNELS, Train_Labels, stratify=Train_Labels, test_size=0.2, random_state=42)
    CH1_train = CH_train[:, 0:h, :]
    CH2_train = CH_train[:, h:2*h, :]
    CH1_val = CH_val[:, 0:h, :]
    CH2_val = CH_val[:, h:2*h, :]

    # Class weights
    sum_2 = np.sum(to_categorical(Train_Labels), axis=0)
    class_weights = assign_weigths(sum_2)

    # Label conversion to one-hot
    CH_trainlabel = to_categorical(CH_trainlabel)
    CH_vallabel = to_categorical(CH_vallabel)

    # Training
    checkpoint_filepath = '25_07_2021_Rats_with_Times_PSD_' + Experiment
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                                   monitor='val_f1_score', mode='max', save_best_only=True)
    stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100, verbose=1, mode='max',
                                               baseline=None, restore_best_weights=True)

    where_am_I = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score', factor=0.1, patience=75, verbose=1, mode='max', min_delta=0.001, cooldown=0, min_lr=0)

    H = model.fit(x=[CH1_train, CH2_train], y=CH_trainlabel, epochs=250, batch_size=64, verbose=2, class_weight=class_weights,
                  validation_data=([CH1_val, CH2_val], CH_vallabel), callbacks=[model_checkpoint_callback, stop_me, where_am_I])

    # Results
    hist_df = pd.DataFrame(H.history)
    pd.DataFrame.from_dict(H.history).to_csv(
        'CSVs//' + checkpoint_filepath, index=False)

    model.save("MODELs\\"+checkpoint_filepath+".h5py")
    loss, recall, accuracy, f1_score = model.evaluate(
        x=[Test_CH1, Test_CH2], y=Test_Labels)

    preds = model.predict(x=[Test_CH1, Test_CH2])
    pred_classes = np.argmax(preds, axis=1)

    Train_CH1, Test_CH1, Train_CH2, Test_CH2, Train_Labels, Test_Labels, Test_Times = create_training_data_PSD(
            animal)

    CONF_MAT = confusion_matrix(Test_Labels,pred_classes)

    ground_truth_in_minutes = []
    true_positive_in_minutes = []
    percentages = []

    # Temporal TP rate calculation for the first 2 hours
    for i in range(2):
        
        Test_Labels = np.transpose(Test_Labels)
        ttimes = Test_Times[i*720:(i+1)*720]
        tlabels = np.transpose(Test_Labels[i*720:(i+1)*720])
        predicted = pred_classes[i*720:(i+1)*720]
        ground_truth = np.sum(ttimes[tlabels == 1])/1000/60
        true_positives = np.sum(
            ttimes[np.logical_and(tlabels == 1, predicted == 1)])/1000/60
        percentage_time = true_positives/ground_truth
        ground_truth_in_minutes.append(ground_truth)
        true_positive_in_minutes.append(true_positives)
        percentages.append(percentage_time)

    # Calculation for the remaining 
    ttimes = Test_Times[1440:-1]
    tlabels = Test_Labels[1440:-1]
    predicted = pred_classes[1440:-1]

    ground_truth = np.sum(ttimes[tlabels == 1])/1000/60
    true_positives = np.sum(
        ttimes[np.logical_and(tlabels == 1, predicted == 1)])/1000/60
    percentage_time = true_positives/ground_truth
    ground_truth_in_minutes.append(ground_truth)
    true_positive_in_minutes.append(true_positives)
    percentages.append(percentage_time)

    TP = int(CONF_MAT[0,0])
    FP = int(CONF_MAT[1,0])
    TN = int(CONF_MAT[1,1])
    FN = int(CONF_MAT[0,1])

    # Save evaluation results to JSON file 
    import json
    name = "JSONs//" + Experiment + "with_conf_mat.json"
    dict1 = {"Loss": loss, "Recall": recall, "Accuracy": accuracy, "F1": f1_score, "GroundTruth_in_Minutes":
             ground_truth_in_minutes, "True_Positive_in_Minutes": true_positive_in_minutes, "Percentages": percentages, "TP":TP, "FP":FP, "TN":TN, "FN":FN, "FP_rate":FP/np.sum(CONF_MAT,axis=None)}
    out_file = open(name, "w")
    json.dump(dict1, out_file, indent=6)
    out_file.close()
