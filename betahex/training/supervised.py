import numpy as np
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn
import tables
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import learn

from betahex.features import Features
from betahex.models import make_policy

tf.logging.set_verbosity(tf.logging.INFO)


def make_train_model(feat):
    p = make_policy(feat, 128, [4, 1])

    def train_model(x, y, mode):
        activation, logits = p(x, mode)

        loss = None
        train_op = None

        first_guess = tf.argmax(input=logits, axis=1, name="guessed_move")
        target = tf.argmax(input=y, axis=1, name="actual_move")
        probabilities = tf.nn.softmax(logits, name="move_probabilities")

        tf.summary.image("probabilities_img", tf.reshape(probabilities, [-1, 19, 13, 1]), max_outputs=6)

        act = tf.reshape(y * 160, [-1, 19, 13, 1])
        black = x['black'] * 220
        white = x['white'] * 200
        empty = x['empty'] * 20
        # ind = np.arange(19 * 13).reshape([1, 19, 13, 1])
        # inds = np.repeat(ind, 128, axis=0)
        # guess = tf.cast(tf.equal(inds, first_guess), tf.int8) * 20

        red = tf.cast(white + empty, tf.uint8)
        green = tf.cast(act + empty, tf.uint8)
        blue = tf.cast(black + empty, tf.uint8)

        tf.summary.image("position_img", tf.concat([red, green, blue], axis=3), max_outputs=6)

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            xent = tf.losses.softmax_cross_entropy(
                onehot_labels=y, logits=logits)

            # raw_output_activation = tf.get_variable('1-filter-conv-output/convolution')
            penal = penalize_invalid(activation, x['empty'], x['ones'], mode)

            loss = xent + 1e-4 * penal
            tf.summary.histogram("guess", first_guess)
            tf.summary.histogram("target", target)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=3e-3,
                optimizer="Adam",
                learning_rate_decay_fn=lambda lr, step: tf.train.exponential_decay(lr, step, 10000, .9)
            )

        # Generate Predictions
        predictions = {
            "classes": first_guess,
            "probabilities": probabilities,
            "logits": logits
        }

        # metrics = {
        #     "accuracy":
        #         learn.MetricSpec(
        #             metric_fn=accuracy, prediction_key="classes")
        # }
        accrcy = tf.reduce_mean(tf.to_float(tf.equal(first_guess, tf.argmax(y, axis=1))))
        tf.summary.scalar("train_accuracy", accrcy)

        tf.train.Scaffold()

        # Return a ModelFnOps object
        return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )

    return train_model


def make_input_fn(feat, data, batch_size):
    # opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    # reader = tf.TFRecordReader(options=opt)

    input_features = {node.name: node for node in data.get_node('/x')}
    ys = np.reshape(data.get_node('/y'), (-1, feat.shape[0] * feat.shape[1]))
    return numpy_input_fn(
        input_features, ys, batch_size=batch_size,
        num_epochs=None, shuffle=True, num_threads=1
    )


def accuracy(labels, predictions, weights=None, metrics_collections=None,
             updates_collections=None, name=None):
    return tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions, weights,
        metrics_collections, updates_collections, name
    )


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


def main(unused_argv):
    # Load training and eval data
    feat = Features(13)

    model_fn = make_train_model(feat)

    config = learn.RunConfig(save_checkpoints_secs=60)

    est = learn.Estimator(
        model_fn=model_fn, model_dir="data/tf/models/128f-4-bn-1-mask-1e-4-penal-1-0-l3e-3d.9adam", config=config
    )

    training_data = tables.open_file('data/hdf5/training.h5')
    train_in = make_input_fn(feat, training_data, 128)

    # training_queue = tf.train.string_input_producer(
    #     ["data/tf/features/training.tfrecords"], num_epochs=5
    # )
    # training_data = None
    # train_in = make_input_fn(feat, training_data, 64)

    # validation_queue = tf.train.string_input_producer(
    #     ["data/tf/features/validation.tfrecords"], num_epochs=10
    # )
    #
    # validation_data = None
    # eval_in = make_input_fn(feat, validation_data, 128)

    validation_data = tables.open_file('data/hdf5/validation.h5')
    eval_in = make_input_fn(feat, validation_data, 64)

    for i in range(50):
        est.fit(
            input_fn=train_in,
            steps=2000
        )
        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }
        est.evaluate(input_fn=eval_in, metrics=metrics, steps=100)

    training_data.close()


if __name__ == '__main__':
    tf.app.run()
