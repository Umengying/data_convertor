import tensorflow as tf 


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=2)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # be careful to keep the dtype of image the same!
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [224,224,3])

    label = tf.cast(features['label'], tf.int8)
    return image, label


def main():
	img, label = read_and_decode('example.tfrecords')
	init_op = tf.group(tf.global_variables_initializer(), 
		tf.local_variables_initializer())
	with tf.Session() as sess:
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		try:
			while not coord.should_stop():
				sess.run([img, label])
				print('ok!')

		except tf.errors.OutOfRangeError:
			print('Done!')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
		# Wait for threads to finish.
		coord.join(threads)


if __name__ == '__main__':
	main()
