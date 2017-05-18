'''
convert image data to standard tensorflow formats
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from PIL import Image


# Input must be type int or long.
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Input must be type bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(examples, name):
    filename = name + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for _, example in enumerate(examples):
        exp = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(example[1]),
            'image_raw': _bytes_feature(example[0])}))
        writer.write(exp.SerializeToString())
    writer.close()


def getExamples(dirname):
    ''' get examples '''
    datalist = []
    imglist = os.walk(dirname).next()[2]
    
    sess = tf.Session()
    for i, imgname in enumerate(imglist):
        animal, _, _ = imgname.split('.')
        imgpath = os.path.join(dirname, imgname)
        # # TODO: change the way of opening image
        # img = tf.image.decode_jpeg(tf.read_file(imgpath), channels=3)
        # # Tensor of 2 elements: `new_height, new_width`
        # resized_image = tf.image.resize_images(img, [224, 224])
        # img_raw = sess.run(tf.cast(resized_image, tf.uint8))
        img = Image.open(imgpath)
        resized_img = img.resize((224, 224))
        img_raw = resized_img.tobytes()
        if animal == 'cat':
            label = 0
        else:
            label = 1
        example = [img_raw, label]
        datalist.append(example)
        print(i, len(datalist))
    sess.close()
    return datalist


def main():
    dirname = 'Dogs_vs_Cats/train'
    examples = getExamples(dirname)
    convert_to(examples, 'train')

if __name__ == '__main__':
    main()
