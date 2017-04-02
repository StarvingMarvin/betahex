import tensorflow as tf
from betahex.models.common import CommonModel


class Policy(CommonModel):

    def __init__(self, board_size=13, input_features=None, layer_dim_5=60, layer_dims_3=None):
        super().__init__(board_size, input_features, layer_dim_5, layer_dims_3)
        self.y = tf.placeholder(tf.float16, [None, self.skewed_dim[0], self.skewed_dim[1]])

    def model(self):

        common = self.get()

        W_out = tf.Variable(
            tf.random_normal([1, 1, self.layer_dims_3[-1], 1], dtype=tf.float16),
            dtype=tf.float16, name='Policy_W_out'
        )
        activation = tf.nn.conv2d(common, W_out, strides=[1, 1, 1, 1],
                           padding='VALID', name='Policy_activation')

        out = tf.nn.softmax(activation, name='Policy_softmax')

        return out

