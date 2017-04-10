import tensorflow as tf


def conv_layer(x, W, b, name=None):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b, name=name)
    return conv_out


class CommonModel:

    def __init__(self, features, layer_dim_5=60, layer_dims_3=None):
        self.layer_dim_5 = layer_dim_5
        self.layer_dims_3 = layer_dims_3 or [60] * 6
        self.features = features

        self.x = tf.placeholder(
            tf.float16,
            (None,) + features.shape,
            'x'
        )

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get(self):
        h1_size = self.layer_dim_5
        W1 = tf.Variable(tf.random_normal([5, 5, self.features.shape[2], h1_size], dtype=tf.float16), dtype=tf.float16)
        b1 = tf.Variable(tf.random_normal([h1_size], dtype=tf.float16), dtype=tf.float16)
        padded = tf.pad(self.x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        prev = conv_layer(padded, W1, b1)

        prev_size = h1_size

        for dim in self.layer_dims_3:
            p = tf.pad(prev, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            w = tf.Variable(tf.random_normal([3, 3, prev_size, dim], dtype=tf.float16), dtype=tf.float16)
            b = tf.Variable(tf.random_normal([dim], dtype=tf.float16), dtype=tf.float16)
            prev = conv_layer(p, w, b)
            prev_size = dim

        return prev
