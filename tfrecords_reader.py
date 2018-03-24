import tensorflow as tf
import numpy as np


class Reader(object):
    def __init__(self, tfrecords_files, batch_size=1, patch_size=24, scale=1,
                 min_queue_examples=1000, epochs=100, num_threads=8, return_LR_same_size_with_HR=False,
                 patch_augmentation_options=None, name=''):
        self.tfrecords_files = tfrecords_files
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scale = scale
        self.min_queue_examples = min_queue_examples
        self.num_threads = num_threads
        self.return_LR_same_size_with_HR = return_LR_same_size_with_HR
        self.patch_augmentation_options = patch_augmentation_options
        self.reader = tf.TFRecordReader()
        self.name = name
        self.filename_queue = tf.train.string_input_producer(self.tfrecords_files, num_epochs=epochs)

    def feed(self):
        with tf.name_scope(self.name):
            _, serialized_example = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'HR_file_name': tf.FixedLenFeature([], tf.string),
                    'LR_file_name': tf.FixedLenFeature([], tf.string),
                    'HR_image': tf.FixedLenFeature([], tf.string),
                    'LR_image': tf.FixedLenFeature([], tf.string),
                    'HR_image_width': tf.FixedLenFeature([], tf.int64),
                    'HR_image_height': tf.FixedLenFeature([], tf.int64),
                    'LR_image_width': tf.FixedLenFeature([], tf.int64),
                    'LR_image_height': tf.FixedLenFeature([], tf.int64)
                })

            HR_image_width = tf.cast(features['HR_image_width'], tf.int32)
            HR_image_height = tf.cast(features['HR_image_height'], tf.int32)
            LR_image_width = tf.cast(features['LR_image_width'], tf.int32)
            LR_image_height = tf.cast(features['LR_image_height'], tf.int32)

            HR_image = tf.image.decode_jpeg(features['HR_image'], channels=3)
            HR_image = tf.reshape(HR_image, [HR_image_height, HR_image_width, 3])
            HR_image = tf.image.convert_image_dtype(HR_image, dtype=tf.float32)

            LR_image = tf.image.decode_jpeg(features['LR_image'], channels=3)
            LR_image = tf.reshape(LR_image, [LR_image_height, LR_image_width, 3])
            LR_image = tf.image.convert_image_dtype(LR_image, dtype=tf.float32)

            random_height_start = tf.random_uniform([], minval=0, maxval=LR_image_height - self.patch_size,
                                                    dtype=tf.int32)
            random_width_start = tf.random_uniform([], minval=0, maxval=LR_image_width - self.patch_size,
                                                   dtype=tf.int32)

            HR_image = HR_image[self.scale*random_height_start:self.scale*(random_height_start + self.patch_size),
                                self.scale*random_width_start: self.scale*(random_width_start + self.patch_size)]
            LR_image = LR_image[random_height_start:random_height_start + self.patch_size,
                                random_width_start:random_width_start + self.patch_size]

            HR_image = tf.random_crop(HR_image, [self.patch_size*self.scale, self.patch_size*self.scale, 3])
            LR_image = tf.random_crop(LR_image, [self.patch_size, self.patch_size, 3])

            if self.patch_augmentation_options["flip_and_rotate"] is True:
                random_flip_var = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
                HR_image = tf.cond(random_flip_var < 0.5, lambda: HR_image, lambda: tf.image.flip_left_right(HR_image))
                LR_image = tf.cond(random_flip_var < 0.5, lambda: LR_image, lambda: tf.image.flip_left_right(LR_image))

                random_rotate_var = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
                random_rotate_var = tf.cast(random_rotate_var, tf.float32)
                HR_image = tf.contrib.image.rotate(HR_image, (np.pi / 2) * random_rotate_var)
                LR_image = tf.contrib.image.rotate(LR_image, (np.pi / 2) * random_rotate_var)

            HR_images, LR_images = tf.train.shuffle_batch(
                [HR_image, LR_image], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3*self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )

        return HR_images, LR_images


def test_reader():
    TRAIN_FILE = 'dataset/2017/train_2017_bicubic_X4.tfrecords'

    with tf.Graph().as_default():
        reader = Reader(TRAIN_FILE, batch_size=2, scale=4)
        HR_images, LR_images = reader.feed()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                np_HR_images, np_LR_images = sess.run([HR_images, LR_images])
                print(np_HR_images)
                print("HR_images shape: {}".format(np_HR_images.shape))
                print("LR_images shape: {}".format(np_LR_images.shape))
                print("="*10)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
