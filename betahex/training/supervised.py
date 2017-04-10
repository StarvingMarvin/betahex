import tensorflow as tf
import numpy as np
import tables

from betahex.models.features import Features
from betahex.models.policy import Policy


if __name__ == '__main__':
    feat = Features(13)
    p = Policy(feat)
    data = tables.open_file('data/hdf5/sample.h5')

    input_features = {node.name: node for node in data.get_node('/x')}

    ys = np.reshape(data.get_node('/y'), (-1, feat.shape[0] * feat.shape[1]))
    y = tf.placeholder(tf.float16, [None, feat.shape[0] * feat.shape[1]])

    input_vector = feat.combine(input_features)
    model = p.model()
    out = tf.nn.softmax_cross_entropy_with_logits(
        logits=model, labels=y, name='Policy_softmax'
    )

    cost = tf.reduce_mean(out)
    train_op = tf.train.AdamOptimizer().minimize(cost)

    guess = tf.argmax(model, 1)
    correct = tf.argmax(y, 1)

    # correct_pred = tf.equal(guess, correct)

    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("shape", feat.shape)
    print("input", {k: np.shape(v) for k, v in input_features.items()})
    print("combined", np.shape(input_vector))
    print("output", np.shape(ys))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dataset_size = np.shape(input_vector)[0]
    for i in range(0, 10):
        print("Epoch:", i)
        for j in range(0, dataset_size, 1000):
            limit = min(dataset_size, j + 1000)
            res, g, c = sess.run(
                [model, guess, correct],
                feed_dict={p.x: input_vector[j:limit], y: ys[j:limit]})
            print(min(g), min(c))
            print(max(g), max(c))
