import tensorflow as tf
from betahex.models.common import CommonModel


class Policy(CommonModel):

    def __init__(self, features, layer_dim_5=60, layer_dims_3=None):
        super().__init__(features, layer_dim_5, layer_dims_3)
        self.y = tf.placeholder(tf.float16, [None, features.shape[0] * features.shape[1]])

    def model(self):

        common = self.get()
        previous_dim = self.layer_dims_3[-1]

        W_out = tf.Variable(
            tf.random_normal([1, 1, previous_dim, 1], dtype=tf.float16),
            dtype=tf.float16, name='Policy_W_out'
        )
        activation = tf.nn.conv2d(common, W_out, strides=[1, 1, 1, 1],
                           padding='VALID', name='Policy_activation')

        logits = tf.reshape(activation, [-1, self.features.shape[0] * self.features.shape[1]])

        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y, name='Policy_softmax')

        return out

