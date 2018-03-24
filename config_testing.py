import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_bool("run_all_ckpts", False, "If True, run all model for subdirectory ckpts")
flags.DEFINE_string("results_dir", '/path_to_result/',
                    "Directory for the SR results to be saved")
flags.DEFINE_string("ckpt_dir", '/path_to_ckpt/',
                    "Directory for the trained model checkpoint")
flags.DEFINE_string("LR_dir", "/mnt/nas/Dataset/NTIRE2018/test/parts/DIV2K_test_LR_wild_another_another/",
                    "Directory for the png files of LR images")
flags.DEFINE_bool("is_geometric_self_ensemble", True,
                  "Geometric Self-ensemble")

# Don't touch below
""""""
flags.DEFINE_string("model", None,
                    "[SRCNN, SRDenseNet, SRDenserNet, SRDenserNet_v2, SRDenserNet_v3, SRDenserNet_v4]")
flags.DEFINE_bool("return_LR_same_size_with_HR", None, "")
# Manifold
flags.DEFINE_bool("is_haar", None, "To use manifold of Haar wavelet space")
flags.DEFINE_bool("is_residual", None, "To use residual learning")
flags.DEFINE_integer("scale", None, "Image scale from LR to HR")
""""""

FLAGS = flags.FLAGS
