import numpy as np
import tensorflow as tf
import sys

from collections import OrderedDict
from functools import wraps

from betahex.game import Move, distances as dist
from betahex.utils import parametrized


def shape_feature(f):
    f.cache = {}

    @wraps(f)
    def wrap(board):
        cached = f.cache.get(board.shape)
        if not cached:
            cached = f(board)
            f.cache[board.shape] = cached
        return cached

    return wrap


@parametrized
def feature(f, depth=1):
    f.is_feature = True
    f.depth = depth
    f.name = f.__name__
    return f


def skewed_dim(n):
    return 3 * (n - 1) // 2 + 1


def skew(features):
    """

    :param features: array-like of shape (w, h, num_of_features)
    :return:
    """
    w, h, d = np.shape(features)
    dims = [skewed_dim(w), h, d]
    skewed = np.zeros(dims, dtype=np.uint8)

    ys = np.repeat(np.tile(np.arange(w), h), d)
    xs = np.repeat(np.arange(h), w * d) + ys // 2
    zs = np.tile(np.arange(d), w * h)

    idx = np.ravel_multi_index(
        [xs, ys, zs],
        dims
    )

    np.put(skewed, idx, features.flat)

    return skewed


@feature
@shape_feature
def black_edges(board):
    w, h = board.shape()
    return np.fromfunction(
        lambda x, y: (y == 0) + (y == h - 1),
        (w, h),
        dtype=np.int8
    )


@feature
@shape_feature
def white_edges(board):
    w, h = board.shape()
    return np.fromfunction(
        lambda x, y: (x == 0) + (x == w - 1),
        (w, h),
        dtype=np.int8
    )


@feature
@shape_feature
def ones(board):
    return np.ones(board.shape(), np.int8)


@feature
def black(board):
    return board.colors() == Move.B


@feature
def white(board):
    return board.colors() == Move.W


@feature
def empty(board):
    return board.colors() == 0


@feature(depth=6)
def recentness(board):
    moves = board.move_numbers()
    latest = np.amax(moves)
    if latest == 0:
        return np.zeros(tuple(board.shape()) + (6,), dtype=np.int8)

    return np.dstack((
        moves == latest,
        (max(latest - 2, 1) <= moves) & (moves < latest),
        (max(latest - 6, 1) <= moves) & (moves < latest - 2),
        (max(latest - 14, 1) <= moves) & (moves < latest - 6),
        (max(latest - 30, 1) <= moves) & (moves < latest - 14),
        moves < latest - 30,
    ))


@feature(depth=40)
def distances(board):
    dist_n = dist(board, 'N')
    dist_s = dist(board, 'S')
    dist_e = dist(board, 'E')
    dist_w = dist(board, 'W')

    def dist_to_feat(dist, mask):
        return np.dstack((mask & (dist == 0),
                          mask & (dist == 1),
                          mask & (dist == 2),
                          mask & (dist == 3),
                          mask & (dist == 4),
                          mask & (dist == 5),
                          mask & (dist == 6),
                          mask & (dist == 7),
                          mask & (dist >= 8) & (dist < 16),
                          mask & (dist >= 16)))

    b = board.colors() == Move.B
    w = board.colors() == Move.W

    return np.dstack(
        (dist_to_feat(dist_n, b),
         dist_to_feat(dist_s, b),
         dist_to_feat(dist_w, w),
         dist_to_feat(dist_e, w))
    )


FEATURES = OrderedDict()

for name in sorted(dir(sys.modules[__name__])):
    f = getattr(sys.modules[__name__], name)
    if getattr(f, 'is_feature', False):
        FEATURES[f.__name__] = f


class Features:
    def __init__(self, board_size, feature_names=None):
        self.feature_names = feature_names or FEATURES.keys()
        self.board_size = board_size
        self.shape = (
            skewed_dim(board_size),
            board_size,
            sum(FEATURES[name].depth for name in self.feature_names)
        )

    def input_vector(self, board):
        return skew(
            np.dstack(
                [FEATURES[feature](board) for feature in self.feature_names],
            )
        )

    def input_map(self, board):
        return self.split(np.asarray(self.input_vector(board), np.float32))

    def input_example(self, board, move):
        feat_map = {
            k: tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=np.asarray(v, dtype=np.float32).flatten()
                )
            )
            for k, v in self.input_map(board).items()
        }

        y = np.asarray(self.one_hot_move(move), dtype=np.int64)
        feat_map['y'] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=y.flatten())
        )

        return tf.train.Example(features=tf.train.Features(feature=feat_map))

    def split(self, input_vector):
        dims = np.ndim(input_vector)
        if 4 < dims < 3:
            raise ValueError("Input vector must have 3 or 4 dimensions, got %s" % dims)
        if dims == 3:
            input_vector = np.reshape(input_vector, (1,) + tuple(np.shape(input_vector)))

        res = {}
        z = 0

        for name in self.feature_names:
            feature = FEATURES[name]
            depth = feature.depth
            res[name] = input_vector[:, :, :, z:z + depth]
            z += depth
        return res

    def combine(self, feature_map):
        vals = [feature_map[name] for name in self.feature_names]
        return np.concatenate(vals, 3)

    def one_hot_move(self, move):
        one_hot = np.fromfunction(
            lambda x, y: (y == move.y) & (x == move.x + move.y // 2),
            self.shape[0:2]
        )

        return one_hot

    def dimension(self, feature_name):
        return self.surface() * FEATURES[feature_name].depth

    def surface(self):
        return self.shape[0] * self.shape[1]

    def feature_shape(self, feature_name):
        return self.shape[0], self.shape[1], FEATURES[feature_name].depth
