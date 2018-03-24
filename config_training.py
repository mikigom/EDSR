import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("restore_dir", None,
                    "Checkpoint directory for re-training")

# Model and Dataset selection
flags.DEFINE_string("model", "EDSR",
                    "[SRCNN, SRDenseNet, SRDenserNet, SRDenserNet_v2, SRDenserNet_v3, SRDenserNet_v4]")
flags.DEFINE_integer("size_of_training_set", 800, "Total size of training pair")
flags.DEFINE_integer("n_epoch", 6000, "Set None for training infinitely")
flags.DEFINE_integer("n_batch_size", 16, "Batch size to train")
flags.DEFINE_integer("n_patch_size", 20, "Patch size to train")
flags.DEFINE_integer("scale", 8, "Image scale from LR to HR")
flags.DEFINE_string("train_tfrecords", 'path_to/train_2018_bicubic_x8.tfrecords',
                    "Which training data to train")

# Loss
flags.DEFINE_bool("is_charbonnier", True, "Using Charbonnier loss")
flags.DEFINE_float("eps_charbonnier", 1e-3, "Hyper-parameter used in Charbonnier loss")

# Augmentation
flags.DEFINE_bool("is_rotate_and_flip", True, "Rotate and Flip data augmentation")
flags.DEFINE_bool("is_adjust_brightness", False, "Not yet implemented")
flags.DEFINE_bool("is_adjust_contrast", False, "Not yet implemented")
flags.DEFINE_bool("is_adjust_gamma", False, "Not yet implemented")
flags.DEFINE_bool("is_adjust_hue", False, "Not yet implemented")
flags.DEFINE_bool("is_adjust_saturation", False, "Not yet implemented")

# Learning Rate
flags.DEFINE_float("learning_rate_exp_decay_rate", 0.5, "Exponential scale of learning rate decay")
flags.DEFINE_integer("learning_rate_exp_decay_epoch", 4000, "")
flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate of optimizer")

# Save model checkpoints periodically
flags.DEFINE_integer("save_step", 5000, "Number of step for saving checkpoints")
flags.DEFINE_bool("store_all_ckpts_periodically", False,
                  "If True, model checkpoints are stored at each save_step. This is usual for validation.")

# Don't touch the below flags.
# These would be set automatically
flags.DEFINE_bool("return_LR_same_size_with_HR", None, "")
FLAGS = flags.FLAGS
