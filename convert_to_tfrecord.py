import tensorflow as tf 
from PIL import Image
import numpy as np

# Input must be type int or long.
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Input must be type bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data, name):
    filename = name + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    # when there is one picture
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(data[1]),
        'image_raw': _bytes_feature(data[0])}))

    writer.write(example.SerializeToString())

    writer.close()


def main():
	img = Image.open('test.jpg')
	resized_img = img.resize((224,224))
	imgbytes = resized_img.tobytes()
	label = 0
	data = [cc, label]
	convert_to(data, 'example')


if __name__ == '__main__':
	main()