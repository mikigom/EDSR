import tensorflow as tf
import random
import os

from PIL import Image

from os import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('HR_dir', 'path_to_HR_data',
                       'HR directory')
tf.flags.DEFINE_string('LR_dir', 'path_to_LR_data',
                       'LR directory')
tf.flags.DEFINE_integer('scale', None, 'scale between HR and LR')
tf.flags.DEFINE_string('output_file', 'path_to_tfrecords/train_2018_bicubic_X8.tfrecords',
                       'output tfrecords file, default: data/test.tfrecords')


def data_reader(input_dir, shuffle=True):
    """Read images from input_dir then shuffle them
    Args:
      input_dir: string, path of input dir, e.g., /path/to/dir
    Returns:
      file_paths: list of strings
    """
    file_paths = []

    for img_file in scandir(input_dir):
        if img_file.name.endswith('.png') and img_file.is_file():
            file_paths.append(img_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        file_paths = [file_paths[i] for i in shuffled_index]

    return file_paths


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(HR_path, HR_image,
                        LR_path, LR_image,
                        HR_image_width, HR_image_height,
                        LR_image_width, LR_image_height):
    HR_file_name = os.path.basename(HR_path)
    LR_file_name = os.path.basename(LR_path)

    example = tf.train.Example(features=tf.train.Features(feature={
        'HR_file_name': _bytes_feature(tf.compat.as_bytes(HR_file_name)),
        'LR_file_name': _bytes_feature(tf.compat.as_bytes(LR_file_name)),
        'HR_image': _bytes_feature(HR_image),
        'LR_image': _bytes_feature(LR_image),
        'HR_image_width': _int64_feature(HR_image_width),
        'HR_image_height': _int64_feature(HR_image_height),
        'LR_image_width': _int64_feature(LR_image_width),
        'LR_image_height': _int64_feature(LR_image_height)
    }))
    return example


def load_paths_from_dir(directory):
    file_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(path, name))

    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    return file_list


def data_writer(HR_dir, LR_dir, output_file):
    """Write data to tfrecords
    """
    HR_filepaths = load_paths_from_dir(HR_dir)
    LR_filepaths = load_paths_from_dir(LR_dir)

    assert len(HR_filepaths) == len(LR_filepaths)
    images_num = len(HR_filepaths)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    coder = ImageCoder()

    for i in range(images_num):
        HR_file_path = HR_filepaths[i]
        LR_file_path = LR_filepaths[i]

        with tf.gfile.FastGFile(HR_file_path, 'rb') as f:
            HR_image_data = f.read()
        with tf.gfile.FastGFile(LR_file_path, 'rb') as g:
            LR_image_data = g.read()

        HR_image = coder.decode_jpeg(HR_image_data)
        LR_image = coder.decode_jpeg(LR_image_data)

        HR_image, HR_image_data = adjust_size(HR_image)

        HR_image_width, HR_image_height = HR_image.size
        LR_image = Image.fromarray(LR_image)
        LR_image_width, LR_image_height = LR_image.size

        if FLAGS.is_for_same_scale is True:
            HR_image = HR_image.resize((LR_image_width, LR_image_height), resample=Image.BICUBIC)
            HR_image.save('dataset/temp.png')
            with tf.gfile.FastGFile('dataset/temp.png', 'rb') as l:
                HR_image_data = l.read()

        example = _convert_to_example(HR_file_path, HR_image_data, LR_file_path, LR_image_data,
                                      HR_image_width, HR_image_height, LR_image_width, LR_image_height)
        writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("Processed {}/{}.".format(i, images_num))
    print("Done.")
    writer.close()


def adjust_size(HR_image):
    # https://competitions.codalab.org/forums/14755/2250/
    height, width = HR_image.shape[0:2]
    HR_image = HR_image[: height - height % FLAGS.scale,
                        : width - width % FLAGS.scale]
    HR_image = Image.fromarray(HR_image)
    HR_image.save('dataset/temp.png')

    with tf.gfile.FastGFile('dataset/temp.png', 'rb') as f:
        HR_image_data = f.read()
    return HR_image, HR_image_data


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def main(unused_argv):
    print("Convert data to tfrecords...")
    data_writer(FLAGS.HR_dir, FLAGS.LR_dir, FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
