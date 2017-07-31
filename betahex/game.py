import numpy as np


class Move:

    B = 1
    W = 2

    SWAP_PIECES = 1
    SWAP_SIDES = 2
    RESIGN = 3
    FORFEIT = 4

    SPECIAL = {
        'swap-pieces': SWAP_PIECES,
        'swap-sides': SWAP_SIDES,
        'resign': RESIGN,
        'forfeit': FORFEIT,
    }

    def __init__(self, data):
        self.data = data

    @classmethod
    def make_move(cls, c, n, x, y, spec=0):
        if isinstance(c, str):
            c = color(c)
        if isinstance(spec, str):
            spec = cls.SPECIAL[spec]
        return cls([c, n, x, y, spec])

    @classmethod
    def make_special_move(cls, c, n, spec):
        return cls.make_move(c, n, 0, 0, spec)

    def swap(self):
        return Move([opposite_color(self.color), self.n, self.y, self.x, self.special])

    def rotate(self, board_size):
        bs = board_size - 1
        return Move([self.color, self.n, bs - self.x, bs - self.y, self.special])

    @property
    def color(self):
        return self.data[0]

    @property
    def n(self):
        return self.data[1]

    @property
    def x(self):
        return self.data[2]

    @property
    def y(self):
        return self.data[3]

    @property
    def special(self):
        return self.data[4]

    def __repr__(self):
        return "Move({}, {}, {}, {}, {})".format(
            'B' if self.color == Move.B else 'W',
            self.n,
            self.x,
            self.y,
            'swap' if self.special else None
        )

    def __sgf__(self):
        return "{}[{}{}]".format(
            'B' if self.color == Move.B else 'W',
            chr(ord('a') + self.x),
            self.y + 1
        )


def color(c):
    return Move.B if c[0].upper() == 'B' else Move.W


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
        if move.special == Move.SWAP_PIECES:
            return self.swap()
        shape = np.shape(self.data)
        b = Board(np.copy(self.data))
        idx = np.ravel_multi_index(
            [[move.x, move.x], [move.y, move.y], [0, 1]],
            shape
        )
        np.put(b.data, idx, [move.color, move.n])
        return b

    def swap(self):
        """
        Flips the board along long diagonal and changes colors of all pieces
        :return: New board with flipped position
        """
        colors = self.data[:, :, 0]
        color_swapped = np.dstack([opposite_color(colors), self.data[:, :, 1:]])
        swapped = np.swapaxes(color_swapped, 0, 1)

        return Board(swapped)

    def rotate(self):
        """
        Rotates board up-down and left-right. color remains the same
        :return: New board with rotated position
        """
        return Board(np.fliplr(np.flipud(self.data)))

    def colors(self):
        return self.data[:, :, 0]

    def move_numbers(self):
        return self.data[:, :, 1]

    def shape(self):
        return np.shape(self.data)[:2]

    def get_move(self, x, y):
        return Move(self.data[x, y])

    def __repr__(self):
        return '\n'.join(' ' * i + ' '.join(['.', 'X', 'O'][field[0]] for field in row)
                         for i, row in enumerate(self.data))


def distances(board, edge):
    color = Move.B if edge in ('N', 'S') else Move.W
    opposite = opposite_color(color)
    UNREACHABLE = 126

    colors = board.colors()
    padded = np.pad(colors, [(1, 1), (1, 1)], 'constant', constant_values=[(-1, -1), (-1, -1)])
    padded_shape = np.shape(padded)
    dst_fs = {
        'N': lambda x, y: y != 0,
        'S': lambda x, y: y != padded_shape[1] - 1,
        'E': lambda x, y: x != padded_shape[0] - 1,
        'W': lambda x, y: x != 0
    }

    dist = UNREACHABLE * np.fromfunction(dst_fs[edge], padded_shape, dtype=np.int8)

    prev = UNREACHABLE * np.ones_like(dist)
    while not np.array_equal(prev, dist):
        prev = np.copy(dist)
        mins = np.min(np.dstack([
            prev[1:-1, :-2], prev[2:, :-2],
            prev[:-2, 1:-1],     prev[2:, 1:-1],
                prev[:-2, 2:], prev[1:-1, 2:]
        ]), 2)

        dist[1:-1, 1:-1] = np.clip(
            (colors == 0) * (mins + 1) +
            (colors == color) * mins +
            (colors == opposite) * UNREACHABLE,
            0, UNREACHABLE
        )

    return dist[1:-1, 1:-1]


def victory(board):
    dist_n = distances(board, 'N')
    dist_s = distances(board, 'S')

    if np.any((dist_n == 0) & (dist_s == 0)):
        return Move.B

    dist_w = distances(board, 'W')
    dist_e = distances(board, 'E')

    if np.any((dist_w == 0) & (dist_e == 0)):
        return Move.W

    return 0


class Game:

    def __init__(self, size):
        self.next_color = Move.B
        self.size = size
        self.moves = []
        self.player_black = ''
        self.player_white = ''
        self.rank_black = ''
        self.rank_white = ''
        self.game_info = ''
        self.result = ''
        self.board = Board.make_of_size(size)

    def play_move(self, x, y):
        # TODO: check if move is valid
        self.place_move(self.next_color, x, y)
        self.next_color = opposite_color(self.next_color)

    def place_move(self, color, x, y):
        # TODO: sanitize input
        move = Move.make_move(color, self.move_number() + 1, x, y)
        self._move(move)

    def play_swap_pieces(self):
        # TODO: check if move #2
        self._move(Move.make_special_move(self.next_color, self.move_number() + 1, Move.SWAP_PIECES))

    def play_swap_sides(self):
        # TODO: check if move #2
        self._move(Move.make_special_move(self.next_color, self.move_number() + 1, Move.SWAP_SIDES))

    def winner(self):
        return victory(self.board)

    def _move(self, move):
        self.moves.append(move)
        if move.special == Move.SWAP_PIECES:
            self.board = self.board.swap()
            self.next_color = opposite_color(self.next_color)
        elif move.special == Move.SWAP_SIDES:
            self.player_black, self.player_white = self.player_white, self.player_black
            self.rank_black, self.rank_white = self.rank_white, self.rank_black
        else:
            self.board = self.board.place_move(move)

    def move_number(self):
        return len(self.moves)


def moves2boards(board_size, moves):
    normal = [Board.make_of_size(board_size)]
    flip = [Board.make_of_size(board_size)]
    rot = [Board.make_of_size(board_size)]
    rot_flip = [Board.make_of_size(board_size)]
    for m in moves:
        if m.special == Move.SWAP_PIECES:
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
