import numpy as np
from os.path import join, dirname
from glob import glob

from betahex.game import Board, Move
from betahex.models.features import Features
from betahex.utils.sgf import read_sgf


def feature_pairs(f, board, move):
    ret = {}
    normal = f.split(f.features(board))
    normal['y'] = f.one_hot_move(move)

    for k, v in normal.items():
        ret[k] = [v]

    rot = f.split(f.features(board.rotate()))
    rot['y'] = f.one_hot_move(move.rotate(board.shape()[0]))

    for k, v in rot.items():
        ret[k].append(v)

    return ret


def moves2dataset(board_size, moves):
    prev = Board.make_of_size(board_size)

    f = Features(board_size)
    fmap = {}

    for m in moves:
        if m.special == Move.SWAP_PIECES:
            prev = prev.swap()
            continue
        cur = prev
        y = m
        if m.color == Move.W:
            cur = prev.swap()
            y = m.swap()

        for k, v in feature_pairs(f, cur, y).items():
            acc = fmap.get(k, [])
            acc.extend(v)
            fmap[k] = acc
        prev = prev.place_move(m)

    return {k: np.stack(v) for k, v in fmap.items()}


def games2dataset(board_size, games):
    data = {}
    for moves in games:
        gd = moves2dataset(board_size, moves)
        for k, v in gd.items():
            acc = data.get(k, np.array([]).reshape(0, np.shape(v)[1], np.shape(v)[2]))
            data[k] = np.concatenate([acc, v])
    return data


def process_sgfs(board_size, input_dir, output, output_type='npy'):
    game_paths = glob(input_dir + "/*.sgf")
    games = [read_sgf(path) for path in game_paths]
    dataset = games2dataset(board_size, games)
    if output_type == 'npy':
        for k, v in dataset.items():
            np.save(join(output, k), v)
    elif output_type == 'npz':
        np.savez(output, **dataset)
    else:
        raise ValueError("Invalid output type: %s" % output_type)


if __name__ == '__main__':
    process_sgfs(13, 'data/sgf/train', 'data/npy/train')
