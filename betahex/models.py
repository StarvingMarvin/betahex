from io import StringIO

import tensorflow as tf
from tensorflow.contrib.learn.python import learn


def conv_layer(x, filters, size, activation,
               name=None, bias=True, reg_scale=None, board=None):

    reg = None if reg_scale is None else tf.contrib.layers.l2_regularizer(scale=reg_scale)
    conv = tf.layers.conv2d(
        x, filters, size, activation=activation, padding='same', use_bias=bias,
        name=name, kernel_initializer=tf.random_normal_initializer(),
        kernel_regularizer=reg, bias_regularizer=reg
    )

    if board is not None:
        conv = conv * board

    return conv


def visualize_layer(features, tensor, channels, name='layer_img', cy=8, ln=False):
    ix = features.shape[0] + 2
    iy = features.shape[0] + 2

    cx = channels // cy

    img = tf.slice(tensor, (0, 0, 0, 0), (1, -1, -1, -1))

    if ln:
        min_val = tf.reduce_min(img)
        img = tf.log(img - tf.minimum(min_val, 1) + 1)

    img = tf.reshape(img, (features.shape[0], features.shape[1], channels))

    img = tf.image.resize_image_with_crop_or_pad(img, iy, ix)
    img = tf.reshape(img, (iy, ix, cy, cx))
    img = tf.transpose(img, (2, 0, 3, 1))
    img = tf.reshape(img, (1, cy * iy, cx * ix, 1))
    tf.summary.image(name, img)


def common_model(features, *, filter_count=None, groups=None, reg_scale=None):

    def model(input, mode):
        tensors = [input[feat] for feat in features.feature_names]
        mangled = tf.concat(tensors, 3)
        prev = conv_layer(
            mangled, filter_count, 5, tf.nn.relu, "5-filter-conv-input",
            bias=True, reg_scale=None
        )

        board = input['ones']

        viz_cnt = 0

        head = groups[:-1]
        tail = groups[-1]

        training = mode == learn.ModeKeys.TRAIN
        fc = filter_count
        rs = reg_scale

        for g, group in enumerate(head):
            if group < 0:
                prev = tf.layers.dropout(prev, rate=-group, training=training)
            elif group == 0:
                prev = tf.layers.batch_normalization(
                    prev, axis=3, training=training, name="batch_norm-g{}".format(g)
                )

            elif 0 < group < 1:
                fc = int(filter_count * group)
                prev = conv_layer(prev, fc, 3, tf.nn.relu, "3-filter-conv-{}-g{}".format(fc, g),
                                  bias=False, reg_scale=rs, board=board)
            else:
                for i in range(group):
                    fc = filter_count
                    prev = conv_layer(prev, fc, 3, tf.nn.relu, "3-filter-conv-g{}-{}".format(g, i),
                                      bias=False, reg_scale=rs, board=board)

        for i in range(tail):
            prev = conv_layer(prev, filter_count, 3, tf.nn.relu, "3-filter-conv-{}".format(i),
                              bias=True, reg_scale=rs, board=board)
            visualize_layer(features, prev, filter_count,
                            "{:0=2}-3-filter-conv-{}".format(viz_cnt, i), ln=True)
            viz_cnt += 1
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


def shift_invalid(x, valid, name=None):
    invalid = 1 - valid
    valid_min = tf.reduce_min(x * valid, name="valid_min")
    invalid_max = tf.reduce_max(x * invalid, name="invalid_max")

    tf.summary.scalar("valid_min", valid_min)
    tf.summary.scalar("invalid_max", invalid_max)
    shift = tf.maximum(invalid_max - valid_min, 0)
    shifted = tf.subtract(x, shift * invalid, name)
    return shifted


def skew(out, features):
    board = tf.reshape(out, (-1, features.board_size, features.board_size, 1))
    padded = tf.pad(board, [[0, 0], [0, features.shape[0] - features.shape[1]], [0, 0], [0, 0]])
    y_first = tf.transpose(padded, [2, 1, 0, 3])
    shredded = tf.unstack(y_first)
    skewed_rows = []
    for i, t in enumerate(shredded):
        indices = [(x - i // 2) % features.shape[0] for x in range(features.shape[0])]
        t_indices = tf.constant(indices, dtype=tf.int32)
        skewed_rows.append(tf.gather(t, t_indices))
    skewed = tf.stack(skewed_rows)
    batch_first = tf.transpose(skewed, [2, 1, 0, 3])
    return batch_first


def make_policy(features, filter_count=None, groups=None, reg_scale=None):

    common_f = common_model(
        features,
        filter_count=filter_count,
        groups=groups,
        reg_scale=reg_scale
    )

    def model(input, mode):
        common = common_f(input, mode)

        drop = tf.layers.dropout(common, 0.95)

        activation = conv_layer(drop, 1, 1, None, "1-filter-conv-output", True,
                                reg_scale=reg_scale)

        tf.summary.image("output_activation_img", activation)

        valid = tf.to_float(input['empty'])

        masked = activation * valid
        logits = tf.reshape(
            masked,
            [-1, features.shape[0] * features.shape[1]],
            name="logits"
        )
        tf.summary.image("output_logits_img", masked)

        return activation, logits

    return model


def make_value(features, filter_count, groups, reg_scale=None):
    common_f = common_model(features, filter_count=filter_count, groups=groups, reg_scale=reg_scale)

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


def drops(ds):
    ret = []
    for d in ds:
        ret.extend([1, 0, -d])
    return ret

MODEL = {
    'name': '80f-relu-(1-bn-drop)x-drop1.2.1.4.1-1-drop95',
    'filters': 80,
    'shape': drops([.1, .2, .1, .4, .1]) + [1],
    'features': ['black', 'white', 'empty', 'recentness', 'distances', 'black_edges', 'white_edges', 'ones'],
    'regularization_scale': None
}
