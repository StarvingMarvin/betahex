import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from betahex.models import make_policy


def draw_position(inputs, guess, target):
    black = inputs['black']
    white = inputs['white']
    empty = inputs['empty']
    white_edges = inputs['white_edges']

    actually = tf.to_float(tf.reshape(target, [-1, 19, 13, 1]))
    ind = np.arange(19 * 13).reshape([1, 19, 13, 1])
    guess = tf.cast(tf.equal(ind, tf.reshape(guess, [-1, 1, 1, 1])), tf.float32)

    red = tf.cast(white * 200 + empty * 30 + guess * 100 + white_edges * 50, tf.int8)
    green = tf.cast(actually * 140 + empty * 30, tf.int8)
    blue = tf.cast(black * 220 + empty * 30 + guess * 220, tf.int8)

    img = tf.bitcast(tf.concat([red, green, blue], axis=3), tf.uint8)

    tf.summary.image("position_img", img, max_outputs=6)


def accuracy(labels, predictions, weights=None, metrics_collections=None,
             updates_collections=None, name=None):
    return tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions, weights,
        metrics_collections, updates_collections, name
    )


def make_policy_input_fn(feat, data, batch_size, epochs=None):

    input_features = set(
        [tf.contrib.layers.real_valued_column(
            f, dtype=tf.float32, dimension=feat.dimension(f)
        )
         for f in feat.feature_names]
    )

    input_features.add(
        tf.contrib.layers.real_valued_column(
            'y', dtype=tf.int64, dimension=feat.surface()
        )
    )

    features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    def input_fn():
        feature_map = tf.contrib.learn.read_batch_features(
            file_pattern=data,
            batch_size=batch_size,
            features=features,
            reader=lambda: tf.TFRecordReader(options=options),
            num_epochs=epochs
        )

        target = feature_map.pop("y")

        fm = {k: tf.reshape(v, (-1, ) + feat.feature_shape(k)) for k, v in feature_map.items()}
        return fm, target

    return input_fn


def penalize_invalid(x, valid_moves, valid_board, mode,
                     invalid_penal=1, oob_penal=0, name=None):
    valid_min = tf.reduce_min(x * tf.to_float(valid_moves))
    bigger_mask = tf.where(x > valid_min, tf.ones_like(x), tf.zeros_like(x))
    if mode == learn.ModeKeys.TRAIN:
        oob_penal_mask = (tf.ones_like(x) - tf.to_float(valid_board)) * oob_penal
        invalid_move_mask = (tf.ones_like(x) - tf.to_float(valid_moves)) * invalid_penal

        total_mask = invalid_move_mask + oob_penal_mask
        penals = (x - valid_min) * total_mask * bigger_mask

        penal = tf.reduce_mean(penals, name=name)
        tf.summary.scalar('invalid_penal', penal)
    else:
        penal = tf.zeros([1])
    return penal


def make_train_model(feat, *,
                     policy_filters=64,
                     policy_shape=None,
                     learning_rate=3e-3,
                     learn_rate_decay=.98,
                     optimizer="Adam",
                     regularization_scale=None):
    policy_shape = policy_shape or [2, 3, -0.5, 2]
    p = make_policy(feat, policy_filters, policy_shape, reg_scale=regularization_scale)

    def train_model(x, y, mode):
        activation, logits = p(x, mode)

        train_op = None

        first_guess = tf.argmax(input=logits, axis=1, name="guessed_move")
        probabilities = tf.nn.softmax(logits, name="move_probabilities")

        loss = None

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=y, logits=logits)

            tf.summary.image("probabilities_img", tf.reshape(probabilities, [-1, 19, 13, 1]), max_outputs=6)
            draw_position(x, first_guess, y)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=learning_rate,
                optimizer=optimizer,
                learning_rate_decay_fn=lambda lr, step: tf.train.exponential_decay(
                    lr, step, 10000, learn_rate_decay
                ),
                summaries=['learning_rate', 'loss']
            )

            accrcy = tf.reduce_mean(tf.to_float(tf.equal(first_guess, tf.argmax(y, axis=1))))
            tf.summary.scalar("train_accuracy", accrcy)

        # Generate Predictions
        predictions = {
            "classes": first_guess,
            "probabilities": probabilities
        }

        tf.train.Scaffold()

        # Return a ModelFnOps object
        return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )

    return train_model
