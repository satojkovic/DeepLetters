from os import replace
import pathlib
import tensorflow as tf
import numpy as np
import string
import cv2
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

class MjSynth:
    def __init__(self, data_root, width=128, height=32):
        self.data_root = pathlib.Path(data_root)
        self.width = width
        self.height = height
        self.all_image_paths = self._read_imlist()
        self.annotation_train = self._read_annotation('train')
        self.annotation_test = self._read_annotation('test')
        self.annotation_val = self._read_annotation('val')
        self.num_train_data = len(self.annotation_train)
        self.num_test_data = len(self.annotation_test)
        self.num_val_data = len(self.annotation_val)
        self.char_list = string.ascii_letters + string.digits
        self.max_label_len = 0

    def _read_imlist(self):
        imlist = []
        with open(self.data_root.joinpath('imlist.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                imlist.append(line)
        return imlist

    def _read_annotation(self, suffix):
        annot = []
        with open(self.data_root.joinpath('annotation_' + suffix + '.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                annot.append(line)
        return annot

    def random_choice(self, random_choice_rate=0.02):
        # choose data at random
        y_train = np.random.choice(self.annotation_train,
            int(self.num_train_data * random_choice_rate), replace=False)
        X_train = self._get_image_paths_with_empty_check(y_train)
        y_val = np.random.choice(self.annotation_val,
            int(self.num_val_data * random_choice_rate), replace=False)
        X_val = self._get_image_paths_with_empty_check(y_val)
        y_test = np.random.choice(self.annotation_test,
            int(self.num_test_data * random_choice_rate), replace=False)
        X_test = self._get_image_paths_with_empty_check(y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _get_image_paths_with_empty_check(self, annotations):
        image_paths = []
        for i, annot in enumerate(tqdm(annotations)):
            image_path, _ = annot.split(' ')
            image = cv2.imread(str(self.data_root.joinpath(image_path)))
            if image is None:
                np.delete(annotations, i)
                continue
            image_paths.append(image_path)
        return image_paths

    def _encode(self, txt):
        encoded_txt = []
        for char in txt:
            encoded_txt.append(self.char_list.index(char))
        return encoded_txt

    def _parse_and_encode(self, path):
        path = pathlib.Path(path)
        txt = path.name.split('_')[1]
        return self._encode(txt)

    def _preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [self.height, self.width])
        image /= 255.0
        return image

    def _preprocess_image_cv(self, path):
        image = cv2.imread(str(self.data_root.joinpath(path)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # convert each image of shape (32, 128, 1)
        w, h = image.shape
        if w < self.width:
            image = np.concatenate((image, np.ones((self.width - w, h))*255))
        if h < self.height:
            image = np.concatenate((image, np.ones((self.width, self.height - h))*255), axis=1)
        image = np.expand_dims(image, axis=2)
        return image / 255.0

    def _load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self._preprocess_image(image)

    def create_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_image_ds, train_label_ds = self._create_dataset(X_train, y_train)
        val_image_ds, val_label_ds = self._create_dataset(X_val, y_val)
        test_image_ds, test_label_ds = self._create_dataset(X_test, y_test)

        # padding
        train_label_ds = pad_sequences(train_label_ds, maxlen=self.max_label_len, padding='post', value=len(self.char_list))
        val_label_ds = pad_sequences(val_label_ds, maxlen=self.max_label_len, padding='post', value=len(self.char_list))
        test_label_ds = pad_sequences(test_label_ds, maxlen=self.max_label_len, padding='post', value=len(self.char_list))

        return (train_image_ds, train_label_ds), (val_image_ds, val_label_ds), (test_image_ds, test_label_ds)

    def _create_dataset(self, X, y):
        image_ds = []
        for path in tqdm(X):
            image = self._preprocess_image_cv(path)
            image_ds.append(image)
        label_ds = []
        for path in tqdm(y):
            txt = self._parse_and_encode(path)
            if len(txt) > self.max_label_len:
                self.max_label_len = len(txt)
            label_ds.append(txt)
        return image_ds, label_ds

if __name__ == "__main__":
    mj_synth = MjSynth('mnt/ramdisk/max/90kDICT32px')
    print('Num. of images:', len(mj_synth.all_image_paths))
    print('All Train {} / All Val {} / All Test {}'.format(
        len(mj_synth.annotation_train), len(mj_synth.annotation_val), 
        len(mj_synth.annotation_test))
    )

    X_train, y_train, X_val, y_val, X_test, y_test = mj_synth.random_choice(random_choice_rate=0.01)
    print('Train {} / Val {} / Test {}'.format(len(y_train), len(y_val), len(y_test)))

    train_ds, val_ds, test_ds = mj_synth.create_datasets(X_train, y_train, X_val, y_val, X_test, y_val)
