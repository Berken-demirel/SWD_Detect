import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Dropout, Flatten, BatchNormalization, SeparableConv1D, Add, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy.random import seed

seed(162)
tf.random.set_seed(162)


def inception_module_1(layer_in):
    conv1 = Conv1D(32, 1, padding='same', activation='relu', kernel_initializer='GlorotNormal')(layer_in)
    conv4 = Conv1D(32, 4, padding='same', activation='relu', kernel_initializer='GlorotNormal')(layer_in)
    conv16 = Conv1D(32, 16, padding='same', activation='relu', kernel_initializer='GlorotNormal')(layer_in)
    conv64 = Conv1D(32, 64, padding='same', activation='relu', kernel_initializer='GlorotNormal')(layer_in)
    layer_out = concatenate([conv1, conv4, conv16, conv64], axis=-1)
    x3 = BatchNormalization()(layer_out)
    return x3


def res_net_block1(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x


def res_net_block_trans(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu', padding='same')(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu', padding='same')(input_data)
    x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Add()([x3, input_trans])
    x = Activation('relu')(x4)
    return x


def Lead_II_way(lead_II):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same')(lead_II)
    layer_out_dropped = Dropout(0.1)(layer_out)
    Batch1 = BatchNormalization()(layer_out_dropped)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    pool1_dropped = Dropout(0.1)(Pool1)
    Incept_1 = inception_module_1(pool1_dropped)
    res2 = res_net_block_trans(Incept_1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # flat = Flatten()(Pool1)
    return Pool1


def Lead_V5_way(lead_V5):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same')(lead_V5)
    layer_out_dropped = Dropout(0.1)(layer_out)
    Batch1 = BatchNormalization()(layer_out_dropped)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    pool1_dropped = Dropout(0.1)(Pool1)
    Incept_1 = inception_module_1(pool1_dropped)
    res2 = res_net_block_trans(Incept_1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    return Pool1


def define_model(in_shape, out_shape):
    input_II = Input(shape=(in_shape, 1))
    input_V5 = Input(shape=(in_shape, 1))
    out_II = Lead_II_way(input_II)
    out_V5 = Lead_V5_way(input_V5)
    layer_out = concatenate([out_II, out_V5], axis=-1)
    sep1 = SeparableConv1D(128, 4, activation='relu', kernel_initializer='GlorotNormal', padding='same')(layer_out)
    flat = Flatten()(sep1)
    Dense_1 = Dense(128, activation='relu')(flat)
    Dropout1 = Dropout(0.5)(Dense_1)
    out = Dense(out_shape, activation='sigmoid')(Dropout1)
    BerkenLeNet = Model(inputs=[input_II, input_V5], outputs=out)
    BerkenLeNet.summary()
    # compile model
    opt = Adam(learning_rate=0.0003)
    BerkenLeNet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['Recall', 'accuracy',
                                                                            tfa.metrics.F1Score(num_classes=2,
                                                                                                threshold=0.5,
                                                                                                average='macro'),
                                                                            tf.keras.metrics.Precision()
                                                                            ])
    return BerkenLeNet
