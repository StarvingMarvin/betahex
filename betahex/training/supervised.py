import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from betahex.features import Features
from betahex.training.common import make_train_model
from betahex.models import MODEL

tf.logging.set_verbosity(tf.logging.INFO)


def make_input_fn(feat, data, batch_size):

    input_features = set([tf.contrib.layers.real_valued_column(f, dtype=tf.int8)
                    for f in feat.feature_names])

    features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

    def input_fn():
        feature_map = tf.contrib.learn.read_batch_record_features(
            file_pattern=data,
            batch_size=batch_size,
            features=features)

        target = feature_map.pop("y")

        return feature_map, target

    return input_fn

    # input_features = {node.name: node for node in data.get_node('/x')}
    # ys = np.reshape(data.get_node('/y'), (-1, feat.shape[0] * feat.shape[1]))
    # return numpy_input_fn(
    #     input_features, ys, batch_size=batch_size,
    #     num_epochs=None, shuffle=True, num_threads=1
    # )


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

    # training_data = tables.open_file('data/hdf5/training.h5')
    # train_in = make_input_fn(feat, training_data, 64)

    # training_queue = tf.train.string_input_producer(
    #     ["data/tf/features/train.tfrecords"], num_epochs=5
    # )
    training_data = None
    train_in = make_input_fn(feat, ["data/tf/features/train.tfrecords"], 64)

    # validation_queue = tf.train.string_input_producer(
    #     ["data/tf/features/eval.tfrecords"], num_epochs=10
    # )

    validation_data = None
    eval_in = make_input_fn(feat, ["data/tf/features/eval.tfrecords"], 128)

    # validation_data = tables.open_file('data/hdf5/validation.h5')
    # eval_in = make_input_fn(feat, validation_data, 64)

    for i in range(10):
        est.fit(
            input_fn=train_in,
            steps=200
        )
        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }
        eval_result = est.evaluate(input_fn=eval_in, metrics=metrics, steps=10)
        if eval_result['accuracy'] < 1e-2 or eval_result['loss'] > 6:
            break

    # training_data.close()
    # validation_data.close()


if __name__ == '__main__':
    tf.app.run()
