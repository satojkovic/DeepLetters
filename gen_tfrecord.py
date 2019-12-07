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
    cls_text = [ann['utf8_string'].encode('utf8')]
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


if __name__ == "__main__":
    args = parse_arguments()
    ct = coco_text.COCO_Text(args.cocotext_json)
    img_ids = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible'), ('class', 'machine printed')]) \
        if args.train_or_val.lower() == 'train' else ct.getImgIds(imgIds=ct.val, catIds=[('legibility', 'legible'), ('class', 'machine printed')])

    writer = tf.python_io.TFRecordWriter(args.output_path)
    for img_id in img_ids:
        img = ct.loadImgs(img_id)
        file_name = img[0]['file_name']
        height = img[0]['height']
        width = img[0]['width']
        train_val_dir = 'train2014' if args.train_or_val else 'val2014'
        with tf.gfile.GFile(os.path.join(args.coco_imgdir, train_val_dir, file_name), 'rb') as fid:
            encoded_jpg = fid.read()

        ann_ids = ct.getAnnIds(img[0]['id'])
        anns = ct.loadAnns(ann_ids)
        for ann in anns:
            tf_example = create_tf_example(ann, file_name, width, height, encoded_jpg)
            writer.write(tf_example.SerializeToString())
    writer.close()
    print('Generated:', args.output_path)