import tensorflow as tf
from tensorflow.contrib.learn.python import learn


def conv_layer(x, filters, size, activation, name=None, bias=True):
    conv = tf.layers.conv2d(
        x, filters, size, activation=activation, padding='same', use_bias=bias,
        name=name, kernel_initializer=tf.random_normal_initializer()
    )
    return conv


def common_model(features, layer_dim_5=None, layer_dims_3=None):

    layer_dims_3 = layer_dims_3 or [128] * 5
    layer_dim_5 = layer_dim_5 or 192

    def model(input, mode):
        tensors = [input[feat] for feat in features.feature_names]
        mangled = tf.cast(tf.concat(tensors, 3), tf.float32)
        prev = conv_layer(mangled, layer_dim_5, 5, tf.nn.elu, None, False)

        for dim in layer_dims_3:
            conv = conv_layer(prev, dim, 3, tf.nn.elu, None, True)
            prev = conv
            # prev = tf.layers.batch_normalization(conv, axis=3, training=mode == learn.ModeKeys.TRAIN)

        prev = tf.layers.batch_normalization(prev, axis=3, training=mode == learn.ModeKeys.TRAIN)
        return prev

    return model


def make_policy(features, layer_dim_5=None, layer_dims_3=None):

    common_f = common_model(features, layer_dim_5, layer_dims_3)

    def model(input, mode):
        common = common_f(input, mode)
        activation = conv_layer(common, 1, 1, None, None, False)
        valid = tf.to_float(input['empty'])
        epsilon = 0.01
        batch_min = tf.reduce_min(activation)
        shifted = activation - batch_min + epsilon
        masked = shifted * valid
        logits = tf.reshape(
            masked,
            [-1, features.shape[0] * features.shape[1]],
            name="logits"
        )

        return logits

    return model

