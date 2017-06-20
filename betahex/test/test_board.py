from betahex.game import *
from nose.tools import *
from os.path import dirname, basename, join


def test_distance():
    b = Board.make_of_size(13).place_move(Move.make_move('B', 1, 2, 0))
    dist = distances(b, 'N')
    eq_(dist[2, 0], 0)

    b = Board.make_of_size(13).place_move(Move.make_move('B', 1, 2, 12))
    dist = distances(b, 'S')
    eq_(dist[2, 12], 0)

    b = Board.make_of_size(13).place_move(Move.make_move('W', 1, 0, 3))
    dist = distances(b, 'W')
    eq_(dist[0, 3], 0)

    b = Board.make_of_size(13).place_move(Move.make_move('W', 1, 12, 3))
    dist = distances(b, 'E')
    eq_(dist[12, 3], 0)


def test_victory():
    b = Board.make_of_size(13).\
        place_move(Move.make_move('B', 1, 2, 0)).\
        place_move(Move.make_move('B', 2, 2, 1)).\
        place_move(Move.make_move('B', 3, 2, 2)).\
        place_move(Move.make_move('B', 4, 2, 3)).\
        place_move(Move.make_move('B', 5, 2, 4)).\
        place_move(Move.make_move('B', 6, 2, 5)).\
        place_move(Move.make_move('B', 7, 2, 6)).\
        place_move(Move.make_move('B', 8, 2, 7)).\
        place_move(Move.make_move('B', 9, 2, 8)).\
        place_move(Move.make_move('B', 10, 2, 9)).\
        place_move(Move.make_move('B', 11, 2, 10)).\
        place_move(Move.make_move('B', 12, 2, 11)).\
        place_move(Move.make_move('B', 13, 2, 12))

    eq_(victory(b), Move.B)

    b = Board.make_of_size(13). \
        place_move(Move.make_move('W', 1, 0, 3)). \
        place_move(Move.make_move('W', 2, 1, 3)). \
        place_move(Move.make_move('W', 3, 2, 3)). \
        place_move(Move.make_move('W', 4, 3, 3)). \
        place_move(Move.make_move('W', 5, 4, 3)). \
        place_move(Move.make_move('W', 6, 5, 3)). \
        place_move(Move.make_move('W', 7, 6, 3)). \
        place_move(Move.make_move('W', 8, 7, 3)). \
        place_move(Move.make_move('W', 9, 8, 3)). \
        place_move(Move.make_move('W', 10, 9, 3)). \
        place_move(Move.make_move('W', 11, 10, 3)). \
        place_move(Move.make_move('W', 12, 11, 3)). \
        place_move(Move.make_move('W', 13, 12, 3))

    eq_(victory(b), Move.W)

    b = Board.make_of_size(13). \
        place_move(Move.make_move('B', 1, 2, 0)). \
        place_move(Move.make_move('W', 2, 1, 3)). \
        place_move(Move.make_move('B', 3, 2, 12))

    eq_(victory(b), 0)


def test_convert():
    from betahex.utils.sgf import read_sgf
    moves = read_sgf(join(dirname(__file__), 'data/1000211.sgf'))
    g = Game(13)
    for m in moves:
        g._move(m)
    print(g.board)
    print(distances(g.board, 'N') * (g.board.colors() == Move.B))
    print(distances(g.board, 'S') * (g.board.colors() == Move.B))
    print(distances(g.board, 'W') * (g.board.colors() == Move.W))
    print(distances(g.board, 'E') * (g.board.colors() == Move.W))
    assert True
