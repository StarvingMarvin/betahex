import tensorflow as tf
from tensorflow.contrib.learn.python import learn


def conv_layer(x, filters, size, activation, name=None, bias=True):
    conv = tf.layers.conv2d(
        x, filters, size, activation=activation, padding='same', use_bias=bias,
        name=name, kernel_initializer=tf.random_normal_initializer()
    )
    return conv


def common_model(features, filter_count=None, groups=None):

    groups = groups or [4, 1]
    filter_count = filter_count or 128

    def model(input, mode):
        tensors = [input[feat] for feat in features.feature_names]
        mangled = tf.cast(tf.concat(tensors, 3), tf.float32)
        prev = conv_layer(mangled, filter_count, 5, tf.nn.elu, "5-filter-conv-input", True)

        head = groups[:-1]
        tail = groups[-1]

        for g, group in enumerate(head):
            for i in range(group):
                prev = conv_layer(prev, filter_count, 3, tf.nn.elu, "3-filter-conv-g{}-{}".format(g, i), True)
            prev = tf.layers.batch_normalization(prev, axis=3, training=mode == learn.ModeKeys.TRAIN,
                                                 name="batch_norm-g{}".format(g))
        for i in range(tail):
            prev = conv_layer(prev, filter_count, 3, tf.nn.elu, "3-filter-conv-{}".format(i), True)
        return prev

    return model


def mask_invalid(x, valid, min_ratio=1e-5, name=None):
    preliminary = tf.multiply(x, valid, name)
    batch_min = tf.reduce_min(preliminary, name="mask_min")
    batch_max = tf.reduce_max(preliminary, name="mask_max")
    gamut = batch_max - batch_min
    epsilon = gamut * min_ratio
    tf.summary.scalar("mask_min", batch_min)
    tf.summary.scalar("mask_max", batch_max)
    tf.summary.scalar("mask_epsilon", epsilon)
    shifted = x - batch_min + epsilon
    masked = tf.multiply(shifted, valid, name)
    return masked


def make_policy(features, filter_count=None, groups=None):

    common_f = common_model(features, filter_count, groups)

    def model(input, mode):
        common = common_f(input, mode)
        activation = conv_layer(common, 1, 1, None, "1-filter-conv-output", False)
        valid = tf.to_float(input['empty'])
        masked = mask_invalid(activation, valid, name="masked-output")
        logits = tf.reshape(
            masked,
            [-1, features.shape[0] * features.shape[1]],
            name="logits"
        )

        return logits

    return model

