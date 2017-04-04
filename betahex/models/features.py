import numpy as np

from betahex.game import Move


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
    latest = np.amax(moves)
    return np.dstack((
        moves == latest,
        np.logical_and(latest - 2 <= moves, moves < latest),
        np.logical_and(latest - 4 <= moves, moves < latest - 2),
        np.logical_and(latest - 8 <= moves, moves < latest - 4),
        np.logical_and(latest - 16 <= moves, moves < latest - 8),
        np.logical_and(latest - 32 <= moves, moves < latest - 16),
    ))


FEATURES = {
    'black': black,
    'white': white,
    'empty': empty,
    'ones': ones,
    'black_edges': black_edges,
    'white_edges': white_edges,
}


class Features:

    def __init__(self, board_size, input_features=None):
        self.input_features = input_features or (
            'black', 'white', 'empty', 'ones', 'black_edges', 'white_edges'
        )
        self.board_size = board_size
        self.shape = (skewed_dim(board_size), board_size)

    def features(self, board):
        return skew(
            np.dstack(
                [FEATURES[feature](board) for feature in self.input_features]
            )
        )

    def split(self, features):
        return {
            feature: features[:, :, idx]
            for idx, feature in enumerate(self.input_features)
        }

    def combine(self, feature_map):
        pass

    def one_hot_move(self, move):
        one_hot = np.fromfunction(
            lambda x, y: np.logical_and(y == move.y, x == move.x + move.y // 2),
            self.shape
        )

        return one_hot

    @property
    def count(self):
        return len(self.input_features)
