import numpy as np
import sys
from functools import wraps

from betahex.game import Move
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

    ys = np.repeat(np.arange(h), w * d)
    xs = np.repeat(np.tile(np.arange(w), h), d) + ys // 2
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
        lambda x, y: (x == 0) + (x == h - 1),
        (w, h),
        dtype=np.int8
    )


@feature
@shape_feature
def white_edges(board):
    w, h = board.shape()
    return np.fromfunction(
        lambda x, y: (y == 0) + (y == w - 1),
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
        np.logical_and(max(latest - 2, 1) <= moves, moves < latest),
        np.logical_and(max(latest - 6, 1) <= moves, moves < latest - 2),
        np.logical_and(max(latest - 14, 1) <= moves, moves < latest - 6),
        np.logical_and(max(latest - 30, 1) <= moves, moves < latest - 14),
        moves < latest - 30,
    ))


FEATURES = {f.name: f for f in
            [getattr(sys.modules[__name__], name)
             for name in dir(sys.modules[__name__])]
            if getattr(f, 'is_feature', False)
            }


class Features:
    def __init__(self, board_size, feature_names=None):
        self.feature_names = feature_names or FEATURES.keys()
        self.board_size = board_size
        self.shape = (skewed_dim(board_size), board_size)

    def input_vector(self, board):
        return skew(
            np.dstack(
                [FEATURES[feature](board) for feature in self.feature_names]
            )
        )

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
        pass

    def one_hot_move(self, move):
        one_hot = np.fromfunction(
            lambda x, y: np.logical_and(y == move.y, x == move.x + move.y // 2),
            self.shape
        )

        return one_hot

    @property
    def depth(self):
        d = 0
        for name in self.feature_names:
            d += FEATURES[name].depth
        return d
