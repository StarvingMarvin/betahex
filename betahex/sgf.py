import re
from betahex.game import Move, color

move_re = re.compile(r'([bwBW])\[((([a-zA-Z])([1-9][0-9]?))|(swap-pieces))\]')


def read_sgf(sgf_path):
    with open(sgf_path) as f:
        return parse_sgf(f.read())


def parse_sgf(sgf_string):
    moves = []
    chunks = sgf_string.split(';')
    n = 0
    for chunk in chunks:
        m = move_re.match(chunk)
        if m:
            c, x, y, special = m.group(1, 4, 5, 6)
            if x:
                x = ord(m.group(4).upper()) - ord('A')
            if y:
                y = int(m.group(5)) - 1
            moves.append(Move(color(c), x, y, special, n))
            n += 1

    return moves
