import cv2
import math
import pytesseract
from imutils.object_detection import non_max_suppression
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

    def loss(self, y_true, y_pred, input_length, label_length):
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def inference(self, x):
        return self.model.predict(x)


class CvEAST:
    def __init__(self, pb_file, width=320, height=320, conf_th=0.5, nms_th=0.4, roi_pad=0.0):
        self.width = width
        self.height = height
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.roi_pad = roi_pad
        self.layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.net = cv2.dnn.readNet(pb_file)
        self.detections = []
        self.confidences = []
        self.tesseract_config = ('-l eng --oem 1 --psm 7')

    def decode(self):
        detections = []
        confidences = []
        height = self.scores.shape[2]
        width = self.scores.shape[3]
        for y in range(0, height):
            scores_data = self.scores[0][0][y]
            x0_data = self.geometry[0][0][y]
            x1_data = self.geometry[0][1][y]
            x2_data = self.geometry[0][2][y]
            x3_data = self.geometry[0][3][y]
            angles_data = self.geometry[0][4][y]
            for x in range(0, width):
                score = scores_data[x]
                if score < self.conf_th:
                    continue
                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles_data[x]
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
                end_x = int(offset_x + (cos_a * x1_data[x]) + (sin_a * x2_data[x]))
                end_y = int(offset_y - (sin_a * x1_data[x]) + (cos_a * x2_data[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                detections.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])
        return [detections, confidences]

    def text_recognition(self, boxes):
        results = []
        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * self.ratio_w)
            start_y = int(start_y * self.ratio_h)
            end_x = int(end_x * self.ratio_w)
            end_y = int(end_y * self.ratio_h)

            # Padding
            pad_x = int((end_x - start_x) * self.roi_pad)
            pad_y = int((end_y - start_y) * self.roi_pad)

            start_x = max(0, start_x - pad_x)
            start_y = max(0, start_y - pad_y)
            end_x = min(self.org_w, end_x + pad_x)
            end_y = min(self.org_h, end_y + pad_y)

            roi = self.org_image[start_y:end_y, start_x:end_x]
            text = pytesseract.image_to_string(roi, config=self.tesseract_config)
            results.append(((start_x, start_y, end_x, end_y), text))
        return sorted(results, key=lambda r:r[0][1])

    def predict(self, image):
        self.org_image = image.copy()
        self.org_h, self.org_w, _ = self.org_image.shape

        self.ratio_h = self.org_h / self.height
        self.ratio_w = self.org_w / self.width

        self.cnn_image = cv2.resize(image, (self.width, self.height))
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (self.width, self.height),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.layer_names)
        self.scores = outs[0]
        self.geometry = outs[1]

        [boxes, confidences] = self.decode()
        boxes = non_max_suppression(np.array(boxes), probs=confidences)
        results = self.text_recognition(boxes)

        return self.ratio_w, self.ratio_h, results