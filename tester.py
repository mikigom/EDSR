import os

import tensorflow as tf
from skimage import io
import numpy as np

import config_helper
import data_generator
from config_testing import FLAGS
from models.EDSR import EDSR

from utils.geometric_self_ensemble import generate_geometric_self_ensemble_examples, \
    merge_results_geometric_self_ensemble

slim = tf.contrib.slim

config_helper.load_flags(FLAGS)


class Tester(object):
    def __init__(self):
        self.define_dataset()
        self.define_model()
        self.define_saver()
        self.ready_for_session()

    def define_dataset(self):
        self.LR_placeholder = tf.placeholder(tf.float32, [None, None, None, 3])
        self.LR = self.LR_placeholder

    def define_model(self):
        self.sr_model = EDSR(self.LR, scale=FLAGS.scale)

        self.model_out = self.sr_model.out

    def define_saver(self):
        self.saver = tf.train.Saver(self.sr_model.var_list)

    def ready_for_session(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

    def test(self):
        if not os.path.exists(FLAGS.results_dir):
            os.makedirs(FLAGS.results_dir)

        try:
            print("[.] Testing Start...")

            images, file_names = data_generator.load_image_from_dir(FLAGS.LR_dir)

            if FLAGS.run_all_ckpts is False:
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                ckpt_directory_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir)

                self.saver.restore(self.sess, ckpt_directory_path)

                self.testing_on_ckpt_CPU(file_names, images, FLAGS.results_dir)


            else:
                self.get_summary_ops()

                ckpt_directory_names = next(os.walk(FLAGS.ckpt_dir))[1]
                for ckpt_directory_name in ckpt_directory_names:
                    ckpt_directory_path = os.path.join(FLAGS.ckpt_dir, ckpt_directory_name)
                    ckpt_directory_path = tf.train.latest_checkpoint(ckpt_directory_path)

                    print("[*] Load checkpoint on {}...".format(ckpt_directory_name))

                    self.sess.run(tf.local_variables_initializer())
                    self.sess.run(tf.global_variables_initializer())

                    self.saver.restore(self.sess, ckpt_directory_path)

                    print("[.] Checkpoint loading ends.")

                    print("[*] Test Running...")

                    results_dir = os.path.join(FLAGS.results_dir, ckpt_directory_name)
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)

                    self.testing_on_ckpt_CPU(file_names, images, results_dir)

                    print("[.] Test ends.")

                    print("[.] Metric ends.")

        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            print('Stop')
            print("[.] Test Complete.")

    def get_summary_ops(self):
        self.ssim_placeholder = tf.placeholder(tf.float32, [])
        self.psnr_placeholder = tf.placeholder(tf.float32, [])
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('psnr', self.psnr_placeholder),
            tf.summary.scalar('ssim', self.ssim_placeholder)
        ])
        self.summary_writer = tf.summary.FileWriter(FLAGS.results_dir)

    def testing_on_ckpt_CPU(self, file_names, images, results_dir):
        for i in range(len(images)):
            image = images[i]

            if FLAGS.is_geometric_self_ensemble is False:
                sr = self.sess.run(self.model_out,
                                   feed_dict={self.LR_placeholder: np.expand_dims(image, axis=0)})

            else:
                self_ensemble_examples = generate_geometric_self_ensemble_examples(image)
                results = []
                for j in range(8):
                    result = self.sess.run(self.model_out,
                                           feed_dict={self.LR: np.expand_dims(self_ensemble_examples[j], axis=0)})

                    results.append(np.squeeze(np.clip(result, 0., 1.)))
                sr = merge_results_geometric_self_ensemble(results)

            sr = np.squeeze(np.clip(sr, 0., 1.))

            io.imsave(results_dir + os.path.basename(file_names[i]), sr)


if __name__ == '__main__':
    tester = Tester()
    tester.test()
    tester.sess.close()
