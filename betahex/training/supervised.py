import numpy as np
import tables
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.monitors import ValidationMonitor
from tensorflow.python.estimator.checkpoint_utils import load_variable

from betahex.features import Features
from betahex.models import make_policy

tf.logging.set_verbosity(tf.logging.INFO)

def make_train_model(feat):
    p = make_policy(feat)

    def train_model(x, y, mode):
        logits = p(x)

        loss = None
        train_op = None

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=y, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.0001,
                optimizer="Adam"
            )

        # Generate Predictions
        predictions = {
            "classes": tf.argmax(
                input=logits, axis=1, name="prediction_class"),
            "probabilities": tf.nn.softmax(
                logits, name="softmax_tensor"),
            "logits": logits
        }

        metrics = {
            "accuracy":
                learn.MetricSpec(
                    metric_fn=accuracy, prediction_key="classes")
        }

        tf.train.Scaffold()

        # Return a ModelFnOps object
        return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )

    return train_model


def make_input_fn(feat, data, batch_size):
    input_features = {node.name: node for node in data.get_node('/x')}
    ys = np.reshape(data.get_node('/y'), (-1, feat.shape[0] * feat.shape[1]))
    return numpy_input_fn(input_features, ys, batch_size=batch_size, num_epochs=5, shuffle=False, num_threads=1)


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
        model_fn=model_fn, model_dir="data/tf/try1", config=config
    )

    training_data = tables.open_file('data/hdf5/training.h5')
    train_in = make_input_fn(feat, training_data, 256)

    # validation_data = tables.open_file('data/hdf5/validation.h5')
    # eval_in = make_input_fn(feat, training_data, 256)

    for i in range(400):
        est.fit(
            input_fn=train_in,
            steps=50
        )

    training_data.close()


if __name__ == '__main__':
    tf.app.run()
