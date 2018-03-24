import tensorflow as tf
slim = tf.contrib.slim

__leaky_relu_alpha__ = 0.2


def bilinear_resize(batches, scale):
    return tf.image.resize_images(batches, (scale*tf.shape(batches)[1], scale*tf.shape(batches)[2]),
                                  method=tf.image.ResizeMethod.BILINEAR, align_corners=True)


def bilinear_x2(batches):
    return bilinear_resize(batches, 2)


def bilinear_x4(batches):
    return bilinear_resize(batches, 4)


def bilinear_x8(batches):
    return bilinear_resize(batches, 8)


def __leaky_relu__(x, alpha=__leaky_relu_alpha__, name='Leaky_ReLU'):
    return tf.maximum(x, alpha*x, name=name)


def res_block(in_, var_scope_name, scale=0.1):
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=256, kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, activation_fn=None)
            x = tf.nn.relu(x)
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def res_block_wide(in_, var_scope_name, mode, scale=0.1):
    is_training = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, activation_fn=None)
            x = tf.nn.relu(x)
            x = slim.dropout(x, 0.7, is_training=is_training)
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def res_block_wide_v2(in_, var_scope_name, mode, scale=0.1):
    is_training = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=256, kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, activation_fn=None)
            x = tf.nn.relu(x)
            x = slim.dropout(x, 0.7, is_training=is_training)
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def res_block_wide_v3(in_, var_scope_name, mode, scale=0.1):
    is_training = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, activation_fn=None)
            x = swish(x)
            x = slim.dropout(x, 0.7, is_training=is_training)
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def res_block_wide_v4(in_, var_scope_name, mode, scale=0.1):
    is_training = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=256, kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, activation_fn=None)
            x = swish(x)
            x = slim.dropout(x, 0.7, is_training=is_training)
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def res_block_crelu(in_, var_scope_name, scale=0.1):
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], kernel_size=3, stride=1, padding='SAME'):
            x = slim.conv2d(in_, num_outputs=64, activation_fn=None)
            x = slim.batch_norm(x)
            x = tf.nn.crelu(x)
            x = slim.conv2d(x, num_outputs=128, activation_fn=None)
            x = slim.batch_norm(x)
            x = x * scale + in_
    return x


def dense_block(in_, var_scope_name):
    def block_conv(block):
        conv = slim.conv2d(block)
        conv_concat = tf.concat([block, conv], axis=3)
        return conv_concat

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d],
                            num_outputs=32, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(in_)
            block1 = conv1
            block2 = block_conv(block1)
            block3 = block_conv(block2)
            block4 = block_conv(block3)
            block5 = block_conv(block4)
            block6 = block_conv(block5)
            block7 = block_conv(block6)
            block8 = block_conv(block7)

    return block8


def dense_res_block(in_, var_scope_name):
    def block_conv(block, linear_term):
        conv = slim.conv2d(tf.pad(block, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT"))
        conv = slim.conv2d(tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT"), activation_fn=None)
        conv = conv + linear_term
        conv_concat = tf.concat([block, conv], axis=3)
        return conv_concat, conv

    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d],
                            num_outputs=32, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu):
            in_ = tf.pad(in_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            conv1 = slim.conv2d(in_)
            block1 = conv1
            block2, conv2 = block_conv(block1, conv1)
            block3, conv3 = block_conv(block2, conv2)
            block4, conv4 = block_conv(block3, conv3)
            block5, conv5 = block_conv(block4, conv4)
            block6, conv6 = block_conv(block5, conv5)
            block7, conv7 = block_conv(block6, conv6)
            block8, _ = block_conv(block7, conv7)

    return block8


def res_dense_block(in_, var_scope_name, scale=0.1):
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d], num_outputs=256, kernel_size=3, stride=1, padding='VALID'):
            x = dense_res_block(in_, 'dense_res')
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, activation_fn=None)
            x = x * scale + in_
    return x


def subpixel_upsample(in_, scale=4, n_feature=256):
    with slim.arg_scope([slim.conv2d], kernel_size=3, stride=1, padding='SAME', activation_fn=None):
        if scale == 4:
            x = slim.conv2d(in_, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
        elif scale == 8:
            x = slim.conv2d(in_, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
        else:
            x = slim.conv2d(in_, n_feature * scale ** 2)
            x = tf.depth_to_space(x, 2)
    return x


def subpixel_upsample_v2(in_, scale=4, n_feature=256):
    with slim.arg_scope([slim.conv2d], kernel_size=3, stride=1, padding='VALID'):
        if scale == 4:
            x = tf.pad(in_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
        elif scale == 8:
            x = tf.pad(in_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = slim.conv2d(x, n_feature * 2 ** 2)
            x = tf.depth_to_space(x, 2)
        else:
            x = slim.conv2d(in_, n_feature * scale ** 2)
            x = tf.depth_to_space(x, 2)
    return x


def subpixel_upsample_once(in_, scale=4):
    with slim.arg_scope([slim.conv2d], kernel_size=3, stride=1, padding='SAME', activation_fn=None):
        x = slim.conv2d(in_, 3 * scale ** 2)
        x = tf.depth_to_space(x, scale)
    return x


def swish(x, beta=1.):
    return x * tf.sigmoid(beta * x)


if __name__ == '__main__':
    print(bilinear_x2(tf.ones([6, 24, 24, 3])).get_shape())
