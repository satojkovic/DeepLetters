import cv2
import math
import numpy as np
import string

from keras.layers import Conv2D, Input, MaxPool2D, BatchNormalization, LSTM, Lambda, Bidirectional, Dense
from keras.models import Model
import keras.backend as K

class CRNN:
    def __init__(self, width=128, height=32):
        self.char_list = string.ascii_letters + string.digits
        self.width = width
        self.height = height
        self.inputs = Input(shape=(self.height, self.width, 1))
        self.conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(self.inputs)
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_1)
        self.conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(self.pool_1)
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_2)
        self.conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(self.pool_2)
        self.conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(self.conv_3)
        self.pool_4 = MaxPool2D(pool_size=(2, 1))(self.conv_4)
        self.conv_5 = Conv2D(512, (3,3), activation='relu', padding='same')(self.pool_4)
        self.batch_norm_5 = BatchNormalization()(self.conv_5)
        self.conv_6 = Conv2D(512, (3,3), activation='relu', padding='same')(self.batch_norm_5)
        self.batch_norm_6 = BatchNormalization()(self.conv_6)
        self.pool_6 = MaxPool2D(pool_size=(2, 1))(self.batch_norm_6)
        self.conv_7 = Conv2D(512, (2,2), activation='relu')(self.pool_6)
        self.squeezed = Lambda(lambda x: K.squeeze(x, 1))(self.conv_7)
        self.blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(self.squeezed)
        self.blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(self.blstm_1)
        self.outputs = Dense(len(self.char_list)+1, activation='softmax')(self.blstm_2)
        self.model = Model(self.inputs, self.outputs)

    def compile(self, max_label_len):
        self.labels = Input(shape=[max_label_len], dtype='float32')
        self.input_length = Input(shape=[1], dtype='int64')
        self.label_length = Input(shape=[1], dtype='int64')
        self.loss_out = Lambda(self._loss, output_shape=(1,), name='ctc')([self.outputs, self.labels, self.input_length, self.label_length])
        self.training_model = Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length], outputs=self.loss_out)
        self.training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    def _loss(self, args):
        y_pred, y_true, input_length, label_length = args
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def predict(self, images):
        preds = self.model.predict(images)
        return K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])

if __name__ == "__main__":
    crnn = CRNN()
    crnn.model.summary()
