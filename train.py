from model import CRNN
from dataset import MjSynth
import argparse
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as K
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='readline')
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, help='Batch size')
    parser.add_argument('--epochs', default=10, help='Num. of epochs')
    parser.add_argument('--save_model_path', default='crnn_model.h5', help='Path to output model')
    parser.add_argument('--debug', action='store_true', help='Invoke tfdbg')
    args = parser.parse_args()

    if args.debug:
        set_debugger_session()

    # Setup dataset
    mj_synth = MjSynth('mnt/ramdisk/max/90kDICT32px')
    print('Num. of images:', len(mj_synth.all_image_paths))
    print('All Train {} / All Val {} / All Test {}'.format(
        len(mj_synth.annotation_train), len(mj_synth.annotation_val), 
        len(mj_synth.annotation_test))
    )

    X_train, y_train, X_val, y_val, X_test, y_test = mj_synth.random_choice(random_choice_rate=0.01)
    print('Train {} / Val {} / Test {}'.format(len(y_train), len(y_val), len(y_test)))

    train_ds, val_ds, test_ds = mj_synth.create_datasets(X_train, y_train, X_val, y_val, X_test, y_val)

    # Model definition
    crnn = CRNN()
    crnn.compile(mj_synth.max_label_len)

    # Train the model
    ckpt = ModelCheckpoint(filepath=args.save_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [ckpt]
    crnn.training_model.fit(
        x=[*train_ds], y=np.zeros(len(train_ds[0])), batch_size=args.batch_size, epochs=args.epochs,
        validation_data=([*val_ds], [np.zeros(len(val_ds[0]))]), verbose=1, callbacks=callbacks_list
    )
