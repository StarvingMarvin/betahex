import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from betahex.features import Features
from betahex.training.common import make_train_model
from betahex.models import MODEL

tf.logging.set_verbosity(tf.logging.INFO)


def make_input_fn(feat, data, batch_size):

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
            reader=lambda: tf.TFRecordReader(options=options)
        )

        target = feature_map.pop("y")

        fm = {k: tf.reshape(v, (-1, ) + feat.feature_shape(k)) for k, v in feature_map.items()}
        return fm, target

    return input_fn


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
        policy_shape=MODEL['shape'],
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

    train_in = make_input_fn(feat, ["data/tf/features/train.tfrecords"], 32)

    eval_in = make_input_fn(feat, ["data/tf/features/eval.tfrecords"], 64)

    for i in range(60):
        est.fit(
            input_fn=train_in,
            steps=3000
        )
        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }
        eval_result = est.evaluate(input_fn=eval_in, metrics=metrics, steps=100)
        if i > 2 and (eval_result['accuracy'] < 1e-2 or eval_result['loss'] > 8):
            break


if __name__ == '__main__':
    tf.app.run()
