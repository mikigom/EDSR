import tensorflow as tf

from models.ops import res_block, subpixel_upsample

slim = tf.contrib.slim


class EDSR(object):
    def __init__(self, in_, scale=4):
        self.in_ = in_
        self.scale = scale

        assert self.scale % 2 == 0

        with tf.variable_scope('EDSR') as vs:
            self.out = self.build_model()
        self.var_list = tf.contrib.framework.get_variables(vs)

    def build_model(self):
        # Preprocessing as mentioned in the paper, by subtracting the mean.
        # But this model isn't implemented.

        with tf.variable_scope('low_level'):
            low_level = slim.conv2d(self.in_, num_outputs=256, kernel_size=3, stride=1, padding='SAME')

        with tf.variable_scope('res_block'):
            block = low_level
            num_layers = 32
            for i in range(num_layers):
                block = res_block(block, 'res_block_'+str(i))

            with tf.variable_scope('res_block_tail'):
                block = slim.conv2d(block, num_outputs=256, kernel_size=3, stride=1)
                block += low_level

        with tf.variable_scope('upsample'):
            up = subpixel_upsample(block, self.scale)

        with tf.variable_scope('reconstruction'):
            reconstruction = slim.conv2d(up, num_outputs=3, kernel_size=3, stride=1, padding='SAME', activation_fn=None)

        return reconstruction


if __name__ == '__main__':
    a = tf.ones([16, 24, 24, 3])
    edsr = EDSR(a, scale=4)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print(sess.run(edsr.out).shape)
    sess.close()
