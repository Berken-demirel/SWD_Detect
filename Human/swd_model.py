import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Dropout, Flatten, BatchNormalization, SeparableConv1D, Add, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy.random import seed

seed(162)
tf.random.set_seed(162)


def res_net_block(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu', padding='same')(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu', padding='same')(input_data)
    x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Add()([x3, input_trans])
    x = Activation('relu')(x4)
    return x


def channel_block(channel):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same')(channel)
    layer_out_dropped = Dropout(0.1)(layer_out)
    Batch1 = BatchNormalization()(layer_out_dropped)
    layer_out_0 = res_net_block(Batch1, 32, 8)
    layer_out_1 = res_net_block(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    pool1_dropped = Dropout(0.1)(Pool1)
    res2 = res_net_block(pool1_dropped, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    return Pool1


def define_model(in_shape, out_shape):
    input_II = Input(shape=(in_shape, 1))
    input_V5 = Input(shape=(in_shape, 1))
    out_II = channel_block(input_II)
    out_V5 = channel_block(input_V5)
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
