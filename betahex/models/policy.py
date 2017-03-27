import tensorflow as tf
import numpy as np
from betahex.game import Board, Move


class Policy:

    def __init__(self, board_size=13, layer_dims=None):
        self.board_size = board_size
        self.layer_dims = layer_dims or [60, 60]
        self.input_features = 6

        self.xdim = 3 * board_size // 2 + 4
        self.ydim = board_size + 4

        self.x = tf.placeholder(tf.float16, [None, self.xdim, self.ydim, self.input_features])
        self.y = tf.placeholder(tf.float16, [None, self.xdim * self.ydim])

    def moves2boards(self, moves):
        normal = [Board.make_of_size(self.board_size)]
        flip = [Board.make_of_size(self.board_size)]
        rot = [Board.make_of_size(self.board_size)]
        rot_flip = [Board.make_of_size(self.board_size)]
        for m in moves:
            if m.special == 'swap-pieces':
                normal, flip = flip, normal
                rot, rot_flip = rot_flip, rot
                continue
            prev = normal[-1]
            cur = prev.place_move(m)
            normal.append(cur)
            flip.append(cur.swap())
            rot.append(cur.rotate())
            rot_flip.append(cur.swap().rotate())
        return normal, flip, rot, rot_flip

    def black_edge(self, x, y):
        bs = self.board_size
        out = np.zeros_like(x)
        np.logical_and(y == 1, 0 <= (x-2), out)
        np.logical_and(out, x-2 < bs, out)
        lower = np.logical_and(y == 2 + bs, 0 <= (x - bs//2 - 2))
        np.logical_and(lower, (x - bs//2 - 2) < bs, lower)
        np.logical_or(out, lower, out)
        return out

    def white_edge(self, x, y):
        bs = self.board_size
        out = np.zeros_like(x)
        np.logical_or(x == y//2, (x - 1 - bs) == y//2, out)
        np.logical_and(out, 0 <= (y - 2), out)
        np.logical_and(out, (y - 2) < bs, out)
        return out

    def board2features(self, board):

        fdim = (self.xdim, self.ydim)
        black = np.zeros(fdim, np.float16)
        white = np.zeros(fdim, np.float16)
        empty = np.zeros(fdim, np.float16)
        ones = np.ones(fdim, np.float16)
        black_edges = np.fromfunction(self.black_edge, fdim)
        white_edges = np.fromfunction(self.white_edge, fdim)

        # TODO: vectorize
        for (x, row) in enumerate(board.data):
            for (y, field) in enumerate(row):
                fx = 2 + x + y // 2
                fy = 2 + y
                black[fx][fy] = field[0] == Move.B
                white[fx][fy] = field[0] == Move.W
                empty[fx][fy] = field[0] == 0

        return np.dstack([black, white, empty, ones, black_edges, white_edges])

    def save(self, path):
        pass

    def load(self, path):
        pass

    def model(self):

        h1_size = self.layer_dims[0]
        W1 = tf.Variable(tf.random_normal([5, 5, self.input_features, h1_size], dtype=tf.float16), dtype=tf.float16)
        b1 = tf.Variable(tf.random_normal([h1_size], dtype=tf.float16), dtype=tf.float16)
        prev = conv_layer(self.x, W1, b1)
        prev_size = h1_size

        for dim in self.layer_dims[1:]:
            p = tf.pad(prev, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            w = tf.Variable(tf.random_normal([3, 3, prev_size, dim], dtype=tf.float16), dtype=tf.float16)
            b = tf.Variable(tf.random_normal([dim], dtype=tf.float16), dtype=tf.float16)
            prev = conv_layer(p, w, b)
            prev_size = dim

        W_out = tf.Variable(tf.random_normal([1, 1, prev_size, 1], dtype=tf.float16), dtype=tf.float16)
        out = tf.nn.conv2d(prev, W_out, strides=[1, 1, 1, 1], padding='VALID')

        return out


def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

