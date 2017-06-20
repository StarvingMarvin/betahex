from io import StringIO

import tensorflow as tf
from tensorflow.contrib.learn.python import learn


def conv_layer(x, filters, size, activation, name=None, bias=True):
    conv = tf.layers.conv2d(
        x, filters, size, activation=activation, padding='same', use_bias=bias,
        name=name, kernel_initializer=tf.random_normal_initializer()
    )
    return conv


def common_model(features, *, filter_count=None, groups=None):

    def model(input, mode):
        tensors = [input[feat] for feat in features.feature_names]
        mangled = tf.cast(tf.concat(tensors, 3), tf.float32)
        prev = conv_layer(mangled, filter_count, 5, tf.nn.relu, "5-filter-conv-input", True)

        head = groups[:-1]
        tail = groups[-1]

        training = mode == learn.ModeKeys.TRAIN

        for g, group in enumerate(head):
            if group < 0:
                prev = tf.layers.dropout(prev, rate=-group, training=training)
            else:
                for i in range(group):
                    fc = 2 * filter_count if i == 0 else filter_count
                    prev = conv_layer(prev, fc, 3, tf.nn.relu, "3-filter-conv-g{}-{}".format(g, i), True)
                prev = tf.layers.batch_normalization(
                    prev, axis=3, training=training, name="batch_norm-g{}".format(g)
                )

        for i in range(tail):
            prev = conv_layer(prev, filter_count, 3, tf.nn.relu, "3-filter-conv-{}".format(i), True)
        return prev

    return model


def mask_invalid(x, valid, name=None):
    batch_min = tf.reduce_min(x, name="mask_min")
    batch_max = tf.reduce_max(x, name="mask_max")

    tf.summary.scalar("mask_min", batch_min)
    tf.summary.scalar("mask_max", batch_max)
    shifted = x - batch_min
    masked = tf.multiply(shifted, valid, name)
    return masked


def make_policy(features, filter_count=None, groups=None):

    common_f = common_model(features, filter_count=filter_count, groups=groups)

    def model(input, mode):
        common = common_f(input, mode)
        activation = conv_layer(common, 1, 1, None, "1-filter-conv-output", False)
        tf.summary.image("output_activation_img", activation)
        valid = tf.to_float(input['empty'])
        masked = mask_invalid(activation, valid, name="masked-output")
        logits = tf.reshape(
            masked,
            [-1, features.shape[0] * features.shape[1]],
            name="logits"
        )
        tf.summary.image("output_logits_img", masked)

        return activation, logits

    return model


def make_value(features, filter_count, groups):
    common_f = common_model(features, filter_count=filter_count, groups=groups)

    def model(input, mode):
        common = common_f(input, mode)
        return common

    return model


def conf2path(filter_count, groups):
    buf = StringIO()
    buf.write(str(filter_count))
    buf.write('f')
    for g in groups:
        if g > 0:
            buf.write("-")
            buf.write(str(g))
    return str(buf)


MODEL = {
    'name': '64f-2-3-drop-5-2-mask-x2-relu',
    'filters': 64,
    'shape': [2, 3, -0.5, 2],
    'features': ['black', 'white', 'empty', 'recentness', 'distances', 'black_edges', 'white_edges', 'ones']
}
