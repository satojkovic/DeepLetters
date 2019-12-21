import sys
sys.path.append('coco-text')
import coco_text
import argparse
import tensorflow as tf
import os
import io
from PIL import Image
from object_detection.utils import dataset_util

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_val', required=True, type=str, help='train or val')
    parser.add_argument('--cocotext_json', required=True, help='Path to cocotext.v2.json')
    parser.add_argument('--coco_imgdir', required=True, help='Path to COCO/images/ directory')
    parser.add_argument('--output_path', required=True, help='Path to output tfrecord')
    return parser.parse_args()

def create_tf_example(ann, file_name, width, height, encoded_jpg):
    x1, y1, w, h = list(map(int, ann['bbox']))
    x2 = x1 + w
    y2 = y1 + h
    xmin = [x1 / width]
    xmax = [x2 / width]
    ymin = [y1 / height]
    ymax = [y2 / height]
    cls_text = ['Text'.encode('utf8')]
    cls_idx = [1] # bbox is 'Text' only, which id is defined in label_map

    filename = file_name.encode('utf8')
    image_format = b'jpg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(cls_text),
        'image/object/class/label': dataset_util.int64_list_feature(cls_idx),        
    }))

    return tf_example

def create_tf_examples(writer, anns, path, file_name, width, height, encoded_jpg):
    xmins, ymins = [], []
    xmaxs, ymaxs = [], []
    classes_text = []
    classes = []
    num_examples = 0
    for ann in anns:
        xmin = ann['bbox'][0]
        ymin = ann['bbox'][1]
        w = ann['bbox'][2]
        h = ann['bbox'][3]
        xmax = xmin + w
        ymax = ymin + h

        # normalize
        xmin /= width
        xmax /= width
        ymin /= height
        ymax /= height

        if xmin < 1 and xmax < 1 and ymin < 1 and ymax < 1:
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append('Text'.encode('utf8'))
            classes.append(1)

    filename = os.path.join(path, file_name)
    filename = filename.encode('utf8')
    image_format = b'jpg'

    if len(xmins) != 0:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        writer.write(tf_example.SerializeToString())
        num_examples += 1
    return num_examples

if __name__ == "__main__":
    args = parse_arguments()
    train_or_val = args.train_or_val.lower()
    ct = coco_text.COCO_Text(args.cocotext_json)
    img_ids = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible')]) \
        if train_or_val == 'train' else ct.getImgIds(imgIds=ct.val, catIds=[('legibility', 'legible')])

    num_examples = 0
    writer = tf.python_io.TFRecordWriter(args.output_path)
    for img_id in img_ids:
        img = ct.loadImgs(img_id)[0]
        file_name = img['file_name']
        train_val_dir = 'train2014'
        path = os.path.join(args.coco_imgdir, train_val_dir)
        pil_img = Image.open(os.path.join(path, file_name))
        width, height = pil_img.size
        # sanity check
        if width != img['width'] or height != img['height']:
            width = img['width']
            height = img['height']
        if width == 0 or height == 0:
            continue

        with tf.gfile.GFile(os.path.join(path, file_name), 'rb') as fid:
            encoded_jpg = fid.read()

        ann_ids = ct.getAnnIds(img['id'])
        anns = ct.loadAnns(ann_ids)
        n = create_tf_examples(writer, anns, path, file_name, width, height, encoded_jpg)
        num_examples += n
    writer.close()
    print('Generated({} examples):'.format(num_examples, args.output_path))