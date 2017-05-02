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
        logits = p(x, mode)

        loss = None
        train_op = None

        first_guess = tf.argmax(input=logits, axis=1, name="guessed_move")
        target = tf.argmax(input=y, axis=1, name="actual_move")

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=y, logits=logits)
            tf.summary.histogram("guess", first_guess)
            tf.summary.histogram("target", target)


        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.01,
                optimizer="Adam",
                learning_rate_decay_fn=lambda lr, step: tf.train.exponential_decay(lr, step, 10000, .9)
            )

        # Generate Predictions
        predictions = {
            "classes": first_guess,
            "probabilities": tf.nn.softmax(
                logits, name="move_probabilities"),
            "logits": logits
        }

        # metrics = {
        #     "accuracy":
        #         learn.MetricSpec(
        #             metric_fn=accuracy, prediction_key="classes")
        # }

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


def main(unused_argv):
    # Load training and eval data
    feat = Features(13)

    model_fn = make_train_model(feat)

    config = learn.RunConfig(save_checkpoints_secs=60)

    est = learn.Estimator(
        model_fn=model_fn, model_dir="data/tf/models/128f-4-bn-1-mask-l01d.9", config=config
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
