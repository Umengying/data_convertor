from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


# build network
def model(input, labels):
    conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5], 
                             padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], 
                             padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    logits = tf.layers.dense(inputs=dropout, units=2)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    total_loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)

    return train_op, total_loss

# read and decode tfrecords
def read_and_decode(filename_queue):
    ''' read and decode tfrecords '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(features['label'], tf.uint8)
    # Random transformations can be put here!
    
    images, labels = tf.train.shuffle_batch([image, label],
                                                 batch_size=10,
                                                 capacity=20,
                                                 num_threads=2,
                                                 min_after_dequeue=0)
    return images, labels


def main():
    ''' main function '''
    filename_queue = tf.train.string_input_producer(['dogs_and_cats.tfrecords'], num_epochs=1)
    images, labels = read_and_decode(filename_queue)
    img = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)
    train_op, loss = model(img, labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                _, lss = sess.run([train_op, loss])
                print(lss)
        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
