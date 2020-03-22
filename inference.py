from model import CRNN
import argparse
import cv2
import numpy as np

def preprocess_image_cv(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    if w < 128:
        image = np.concatenate((image, np.ones((h, 128 - w))*255), axis=1)
    if h < 32:
        image = np.concatenate((image, np.ones((32 - h, 128))*255))
    image = np.expand_dims(image, axis=2)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', required=True, help='Path to input image')
    args = parser.parse_args()

    image = preprocess_image_cv(args.input_image)

    crnn = CRNN()
    crnn.load_weights('crnn_model.h5')
    out = crnn.predict(image)

    for x in out:
        print('predicted: ', end='')
        for p in x:
            if int(p) != -1:
                print(crnn.char_list[int(p)], end='')
        print('')

