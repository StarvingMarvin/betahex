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

    input_vector = feat.combine(input_features)
    model = p.model()

    print("shape", feat.shape)
    print("input", {k: np.shape(v) for k, v in input_features.items()})
    print("combined", np.shape(input_vector))
    print("output", np.shape(ys))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    res = sess.run(model, feed_dict={p.x: input_vector[0:1000], p.y: ys[0:1000]})

    print(np.shape(res))
    print(res[0:100])
