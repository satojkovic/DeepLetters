import cv2
import math

class CvEAST:
    def __init__(self, pb_file, width=320, height=320, conf_th=0.5, nms_th=0.4):
        self.width = width
        self.height = height
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.net = cv2.dnn.readNet(pb_file)
        self.detections = []
        self.confidences = []

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
                offset = ([offset_x + cos_a * x1_data[x] + sin_a * x2_data[x], offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

                p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
                p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))
        return [detections, confidences]

    def predict(self, image):
        org_image = image.copy()
        org_h, org_w, _ = org_image.shape

        self.ratio_h = org_h / self.height
        self.ratio_w = org_w / self.width

        self.cnn_image = cv2.resize(image, (self.width, self.height))
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (self.width, self.height),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.layer_names)
        self.scores = outs[0]
        self.geometry = outs[1]

        [self.boxes, self.confidences] = self.decode()

        indices = cv2.dnn.NMSBoxesRotated(self.boxes, self.confidences, self.conf_th, self.nms_th)

        return self.ratio_w, self.ratio_h, indices, self.boxes