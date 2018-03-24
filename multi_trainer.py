import os
import glob
from shutil import copyfile

import tensorflow as tf
import horovod.tensorflow as hvd
import datetime

from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener

from config_training import FLAGS
from models.EDSR import EDSR
from tfrecords_reader import Reader
import config_helper

slim = tf.contrib.slim

if FLAGS.restore_dir is not None:
    # All flags are over-writen
    restore_dir = FLAGS.restore_dir
    config_helper.load_flags(FLAGS)
    FLAGS.restore_dir = restore_dir


class CheckpointSaverListerner_ckpt_mover(CheckpointSaverListener):
    def __init__(self, ckpt_dir):
        super(CheckpointSaverListerner_ckpt_mover, self).__init__()
        self.ckpt_dir = ckpt_dir

    def begin(self):
        pass

    def before_save(self, session, global_step_value):
        pass

    def after_save(self, session, global_step_value):
        new_ckpt_dir = os.path.join(self.ckpt_dir, str(global_step_value))
        if not os.path.exists(new_ckpt_dir):
            os.makedirs(new_ckpt_dir)

        ckpts_file_list = list()
        ckpts = glob.glob(self.ckpt_dir + 'model.ckpt-' + str(global_step_value) + '*')
        for ckpt in ckpts:
            ckpts_file_list.append(ckpt)
        ckpts_file_list.append(self.ckpt_dir+'checkpoint')
        ckpts_file_list.append(self.ckpt_dir+'graph.pbtxt')

        for file_path in ckpts_file_list:
            copyfile(file_path, os.path.join(new_ckpt_dir, os.path.basename(file_path)))
            if 'model.ckpt-' in file_path:
                os.remove(file_path)

    def end(self, session, global_step_value):
        pass


class Trainer(object):
    def __init__(self):
        hvd.init()

        self.define_dataset()
        self.define_model()
        self.define_loss()
        self.define_optim()
        self.define_writer_and_summary()
        self.define_saver()
        self.initialize_session_and_etc()
        self.define_feed_and_fetch()

    def define_dataset(self):
        patch_augmentation_options = {
            "flip_and_rotate": FLAGS.is_rotate_and_flip,
            "brightness": FLAGS.is_adjust_brightness,
            "contrast": FLAGS.is_adjust_contrast,
            "gamma": FLAGS.is_adjust_gamma,
            "hue": FLAGS.is_adjust_hue,
            "saturation": FLAGS.is_adjust_saturation
        }

        # Tast 1 and 2
        reader = \
            Reader([FLAGS.train_tfrecords],
                   batch_size=FLAGS.n_batch_size,
                   patch_size=FLAGS.n_patch_size,
                   scale=FLAGS.scale,
                   epochs=FLAGS.n_epoch,
                   return_LR_same_size_with_HR=FLAGS.return_LR_same_size_with_HR,
                   patch_augmentation_options=patch_augmentation_options
                   )
        # Task 3 and 4
        """
        reader = \
            Reader(['/path_to_/train_2018_unknown_difficult.tfrecords',
                    '/path_to_/train_2018_unknown_wild_1.tfrecords',
                    '/path_to_/train_2018_unknown_wild_2.tfrecords',
                    '/path_to_/train_2018_unknown_wild_3.tfrecords',
                    '/path_to_/train_2018_unknown_wild_4.tfrecords'],
                   batch_size=FLAGS.n_batch_size,
                   patch_size=FLAGS.n_patch_size,
                   scale=FLAGS.scale,
                   epochs=FLAGS.n_epoch,
                   return_LR_same_size_with_HR=FLAGS.return_LR_same_size_with_HR,
                   patch_augmentation_options=patch_augmentation_options
                   )
        """
        self.HR, self.LR = reader.feed()

    def define_model(self):
        self.sr_model = EDSR(self.LR, scale=FLAGS.scale)

    def define_loss(self):
        diff = self.HR - self.sr_model.out

        if FLAGS.is_charbonnier is True:
            self.loss = tf.reduce_mean(tf.sqrt(tf.square(diff)+FLAGS.eps_charbonnier**2))

        else:
            self.loss = tf.reduce_mean(tf.square(diff))

    def define_optim(self):
        self.step = tf.train.get_or_create_global_step()

        self.learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, self.step,
            FLAGS.learning_rate_exp_decay_epoch*(FLAGS.size_of_training_set/FLAGS.n_batch_size),
            FLAGS.learning_rate_exp_decay_rate,
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = hvd.DistributedOptimizer(optimizer)

        self.opt = optimizer.minimize(self.loss, var_list=self.sr_model.var_list, global_step=self.step)

    def define_writer_and_summary(self):
        def create_1c_image_summary(summary_name, image_1c):
            return tf.summary.image(summary_name, tf.expand_dims(image_1c, axis=-1))

        if FLAGS.restore_dir is None:
            self.ckpt_dir = ''.join(['ckpts/',
                                     FLAGS.model+'_',
                                     str(FLAGS.is_adversarial)+'_',
                                     datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                                     '_'+str(FLAGS.n_patch_size),
                                     '/'])
        else:
            self.ckpt_dir = ''.join(['ckpts/', FLAGS.restore_dir.split('/')[1], '/'])

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.image('LR/image', self.LR),
            tf.summary.image('SR/image', tf.clip_by_value(self.sr_model.out, 0., 1.)),
            tf.summary.image('HR/image', self.HR),
            tf.summary.scalar('learning_rate', self.learning_rate)
        ])

    def define_saver(self):
        self.saver = tf.train.Saver()

    def initialize_session_and_etc(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

        if FLAGS.store_all_ckpts_periodically is False:
            ckpt_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=self.ckpt_dir,
                                                           save_steps=FLAGS.save_step)
        else:
            listern_for_moving_ckpts = CheckpointSaverListerner_ckpt_mover(self.ckpt_dir)
            ckpt_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=self.ckpt_dir,
                                                           save_steps=FLAGS.save_step,
                                                           listeners=[listern_for_moving_ckpts])

        if hvd.rank() == 0:
            hooks = [
                hvd.BroadcastGlobalVariablesHook(0),  # It Must Be
                tf.train.SummarySaverHook(save_steps=50,
                                          output_dir=self.ckpt_dir,
                                          summary_writer=self.summary_writer,
                                          summary_op=self.summary_op),
                ckpt_saver_hook,
            ]
        else:
            hooks = [
                hvd.BroadcastGlobalVariablesHook(0),
            ]

        self.sess = tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks)
        if FLAGS.restore_dir is not None:
            restore_ckpt = tf.train.latest_checkpoint(FLAGS.restore_dir)
            self.saver.restore(self.sess, restore_ckpt)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def define_feed_and_fetch(self):
        self.fetch_dict = {
            "opt": self.opt,
            "loss": self.loss,
        }

    def train(self):
        if hvd.rank() == 0:
            config_dict = config_helper.parse_flags(FLAGS)
            config_helper.save_dict(self.ckpt_dir, config_dict)

        try:
            print("[.] Learning Start...")

            while not self.coord.should_stop():
                self.sess.run(self.fetch_dict)

        except KeyboardInterrupt:
            print("Interrupted")
            self.coord.request_stop()

        finally:
            pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.sess.close()
