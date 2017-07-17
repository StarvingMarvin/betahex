import tensorflow as tf
from tensorflow.contrib import learn

from betahex.features import Features
from betahex.training.common import make_train_model, make_policy_input_fn, accuracy
from betahex.models import MODEL

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    feat = Features(13, MODEL['features'])

    model_fn = make_train_model(
        feat,
        policy_filters=MODEL['filters'],
        policy_shape=MODEL['shape'],
        learning_rate=3e-3,
        learn_rate_decay=.96,
        optimizer="Adam"
    )

    est = learn.Estimator(
        model_fn=model_fn,
        model_dir="data/tf/models/reinforcement/%s" % MODEL['name']
    )

    test_in = make_policy_input_fn(feat, ["data/tf/features/test.tfrecords"], 64, epochs=1)
    eval_in = make_policy_input_fn(feat, ["data/tf/features/eval.tfrecords"], 64, epochs=1)
    sample_in = make_policy_input_fn(feat, ["data/tf/features/sample.tfrecords"], 64, epochs=1)

    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=accuracy, prediction_key="classes")
    }
    est.evaluate(input_fn=eval_in, metrics=metrics)
    est.evaluate(input_fn=test_in, metrics=metrics)
    est.evaluate(input_fn=sample_in, metrics=metrics)


if __name__ == '__main__':
    tf.app.run()
