from collections import namedtuple
import numpy as np

Move = namedtuple('Move', 'color, x, y, special, number')

B = 1
W = 2


def color(c):
    return B if c.upper() == 'B' else W


@np.vectorize
def opposite_color(c):
    return (3 - c) % 3


class Board:

    def __init__(self, data):
        self.data = data

    @classmethod
    def make_of_size(cls, size):
        """
        Creates empty board of given size
        :param size:
        :return:
        """
        return cls(np.zeros([size, size, 2], np.int8))

    def place_move(self, move):
        """
        Creates copy of the board and places a move on it
        :param move:
        :return:
        """
        if move.special == 'swap-pieces':
            return self.flip()
        b = Board(np.copy(self.data))
        idx = np.ravel_multi_index(
            [[move.x, move.x], [move.y, move.y], [0, 1]],
            np.shape(self.data)
        )
        np.put(b.data, idx, [move.color, move.number])
        return b

    def flip(self):
        """
        Flips the board along long diagonal and changes colors of all pieces
        :return: New board with flipped position
        """
        colors, other = np.dsplit(self.data, [1])
        color_swapped = np.dstack([opposite_color(colors), other])
        swapped = np.swapaxes(color_swapped, 0, 1)

        return Board(swapped)

    def rotate(self):
        """
        Rotates board up-down and left-right. color remains the same
        :return: New board with rotated position
        """
        return Board(np.fliplr(np.flipud(self.data)))

    def __repr__(self):
        return '\n'.join(' ' * i + ' '.join(['.', 'X', 'O'][field[0]] for field in row)
                         for i, row in enumerate(self.data))


class Game:

    def __init__(self, size):
        self.next_color = B
        self.size = size
        self.moves = []
        self.player_black = ''
        self.player_white = ''
        self.rank_black = ''
        self.rank_white = ''
        self.game_info = ''
        self.result = ''
        self.board = Board(size)

    def play_move(self, x, y):
        # TODO: check if move is valid
        self.place_move(self.next_color, x, y)
        self.next_color = opposite_color(self.next_color)

    def place_move(self, color, x, y):
        # TODO: sanitize input
        move = Move(color, x, y)
        self.board = self.board.place_move(move)
        self.moves.append(move)

    def play_swap_pieces(self):
        # TODO: check if move #2
        self.moves.append(Move(self.next_color, None, None, 'swap-pieces'))
        self.board = self.board.flip()
        self.next_color = opposite_color(self.next_color)

    def play_swap_sides(self):
        # TODO: check if move #2
        self.player_black, self.player_white = self.player_white, self.player_black
        self.rank_black, self.rank_white = self.rank_white, self.rank_black
        self.moves.append(Move(self.next_color, None, None, 'swap-sides'))
