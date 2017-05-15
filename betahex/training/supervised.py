import numpy as np
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn
import tables
import tensorflow as tf
from tensorflow.contrib import learn

from betahex.features import Features
from betahex.training.common import make_train_model
from betahex.models import MODEL

tf.logging.set_verbosity(tf.logging.INFO)


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
    feat = Features(13, MODEL['features'])

    model_fn = make_train_model(
        feat,
        policy_filters=MODEL['filters'],
        policy_shape=['shape'],
        invalid_penal_weight=1e-5,
        learning_rate=3e-3,
        learn_rate_decay=.96,
        optimizer="Adam"
    )

    config = learn.RunConfig(save_checkpoints_secs=60)

    est = learn.Estimator(
        model_fn=model_fn,
        model_dir="data/tf/models/supervised/%s-1e-5-penal-1-0-l3e-3d.96adamIII" % MODEL['name'],
        config=config
    )

    training_data = tables.open_file('data/hdf5/training.h5')
    train_in = make_input_fn(feat, training_data, 64)

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

    for i in range(100):
        est.fit(
            input_fn=train_in,
            steps=2000
        )
        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }
        eval_result = est.evaluate(input_fn=eval_in, metrics=metrics, steps=100)
        if eval_result['accuracy'] < 1e-2 or eval_result['loss'] > 6:
            break

    training_data.close()
    validation_data.close()


if __name__ == '__main__':
    tf.app.run()
