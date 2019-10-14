import cv2

class CvEAST:
    def __init__(self, pb_file, width=320, height=320):
        self.width = width
        self.height = height
        self.layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.net = cv2.dnn.readNet(pb_file)

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
        (self.scores, self.geometry) = self.net.forward(self.layer_names)