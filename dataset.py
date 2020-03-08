from os import replace
import pathlib
import tensorflow as tf
import numpy as np
import string

class MjSynth:
    def __init__(self, data_root):
        self.data_root = pathlib.Path(data_root)
        self.all_image_paths = self._read_imlist()
        self.annotation_train = self._read_annotation('train')
        self.annotation_test = self._read_annotation('test')
        self.annotation_val = self._read_annotation('val')
        self.num_train_data = len(self.annotation_train)
        self.num_test_data = len(self.annotation_test)
        self.num_val_data = len(self.annotation_val)
        self.char_list = string.ascii_letters + string.digits

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
        X_train = self._get_image_paths(y_train)
        y_val = np.random.choice(self.annotation_val,
            int(self.num_val_data * random_choice_rate), replace=False)
        X_val = self._get_image_paths(y_val)
        y_test = np.random.choice(self.annotation_test,
            int(self.num_test_data * random_choice_rate), replace=False)
        X_test = self._get_image_paths(y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _get_image_paths(self, annotations):
        image_paths = []
        for annot in annotations:
            image_path, _ = annot.split(' ')
            image_paths.append(image_path)
        return image_paths

    def _encode(self, txt):
        encoded_txt = []
        for char in txt:
            encoded_txt.append(self.char_list.index(char))
        return encoded_txt

if __name__ == "__main__":
    mj_synth = MjSynth('mnt/ramdisk/max/90kDICT32px')
    print('Num. of images:', len(mj_synth.all_image_paths))
    print('All Train {} / All Val {} / All Test {}'.format(
        len(mj_synth.annotation_train), len(mj_synth.annotation_val), 
        len(mj_synth.annotation_test))
    )

    X_train, y_train, X_val, y_val, X_test, y_test = mj_synth.random_choice()
    print('Train {} / Val {} / Test {}'.format(len(y_train), len(y_val), len(y_test)))