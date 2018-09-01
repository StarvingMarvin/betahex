import tensorflow as tf

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
        learning_rate=2e-3,
        learn_rate_decay=.98,
        optimizer="Adam",
        regularization_scale=MODEL['regularization_scale']
    )

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        save_summary_steps=100
    )

    est = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="data/tf/models/supervised/%s-l2e-3-d.98adam" % MODEL['name'],
        config=config
    )

    train_in = make_policy_input_fn(feat, ["data/tf/features/train.tfrecords"], 64)

    eval_in = make_policy_input_fn(feat, ["data/tf/features/eval.tfrecords"], 32)

    fouls = 0

    for i in range(40):
        est.train(
            input_fn=train_in,
            steps=2000
        )
        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }
        eval_result = est.evaluate(input_fn=eval_in, metrics=metrics, steps=200)
        if eval_result['accuracy'] < 1e-2 or eval_result['loss'] > 16:
            fouls += 1

        if fouls > 3:
            break


if __name__ == '__main__':
    tf.app.run()
