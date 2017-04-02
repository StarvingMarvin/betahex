import numpy as np
import tensorflow as tf

from betahex.game import Move


def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out


def skewed_dim(n):
    return 3 * (n - 1) // 2 + 1


def skew(features):
    """

    :param features: array-like of shape (w, h, num_of_features)
    :return:
    """
    w, h, d = np.shape(features)
    dims = [skewed_dim(w), h, d]
    skewed = np.zeros(dims)

    ys = np.repeat(np.arange(h), w * d)
    xs = np.repeat(np.tile(np.arange(w), h), d) + ys // 2
    zs = np.tile(np.arange(d), w * h)

    idx = np.ravel_multi_index(
        [xs, ys, zs],
        dims
    )

    np.put(skewed, idx, features.flat)

    return skewed


def black_edges(board):
    w, h = board.shape()
    return np.fromfunction(
        lambda x, y: (x == 0) + (x == h - 1),
        (w, h)
    )


def white_edges(board):
    w, h = board.shape()
    return np.fromfunction(
        lambda x, y: (y == 0) + (y == w - 1),
        (w, h)
    )


def black(board):
    return board.colors() == Move.B


def white(board):
    return board.colors() == Move.W


def empty(board):
    return board.colors() == 0


def ones(board):
    return np.ones(board.shape(), np.float16)


def move_recentness(board):
    moves = board.moves()


class CommonModel:

    def __init__(self, board_size=13, input_features=None, layer_dim_5=60, layer_dims_3=None):
        self.board_size = 13
        self.layer_dim_5 = layer_dim_5
        self.layer_dims_3 = layer_dims_3 or [60] * 6

        self.input_features = input_features or [
            black, white, empty, ones, black_edges, white_edges
        ]

        self.skewed_dim = (skewed_dim(board_size), board_size)

        self.x = tf.placeholder(
            tf.float16,
            [None, self.skewed_dim[0], self.skewed_dim[1], len(self.input_features)],
            'x'
        )

    def board2features(self, board):
        return skew(
            np.dstack(
                [feature(board) for feature in self.input_features]
            )
        )

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get(self):
        h1_size = self.layer_dim_5
        W1 = tf.Variable(tf.random_normal([5, 5, len(self.input_features), h1_size], dtype=tf.float16), dtype=tf.float16)
        b1 = tf.Variable(tf.random_normal([h1_size], dtype=tf.float16), dtype=tf.float16)
        padded = tf.pad(self.x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        prev = conv_layer(padded, W1, b1)
        prev_size = h1_size

        for dim in self.layer_dims_3:
            p = tf.pad(prev, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            w = tf.Variable(tf.random_normal([3, 3, prev_size, dim], dtype=tf.float16), dtype=tf.float16)
            b = tf.Variable(tf.random_normal([dim], dtype=tf.float16), dtype=tf.float16)
            prev = conv_layer(p, w, b)
            prev_size = dim

        return prev
