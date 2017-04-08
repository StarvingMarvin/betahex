from threading import Thread

import numpy as np
import multiprocessing as mp
from os.path import join
from glob import glob

from betahex.game import Board, Move
from betahex.models.features import Features
from betahex.utils.sgf import read_sgf


def feature_pairs(f, board, move):
    ret = {}
    normal_x = f.input_vector(board)
    normal_y = f.one_hot_move(move)

    rot_x = f.input_vector(board.rotate())
    rot_y = f.one_hot_move(move.rotate(board.shape()[0]))

    ret['x'] = np.stack([normal_x, rot_x])
    ret['y'] = np.stack([normal_y, rot_y])

    return ret


def moves2dataset(f, moves):
    prev = Board.make_of_size(f.board_size)

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
            acc.append(v)
            fmap[k] = acc
        prev = prev.place_move(m)
    ret = {k: np.concatenate(v) for k, v in fmap.items()}

    return ret


def games2dataset(games):
    data = {}
    for moves in games:
        gd = moves2dataset(games2dataset.f, moves)
        for k, v in gd.items():
            acc = data.get(k, [])
            acc.append(v)
            data[k] = acc
    ret = {k: np.concatenate(v) for k, v in data.items()}

    games2dataset.q.put(ret)


def processor_init(f, q):
    games2dataset.f = f
    games2dataset.q = q


def process_sgfs(board_size, input_dir, output_dir):

    q = mp.JoinableQueue()
    feat = Features(board_size, ('black', 'white', 'empty', 'recentness'))

    t = Thread(
        target=writer,
        kwargs={'q': q, 'features': feat, 'output_dir': output_dir}
    )

    t.daemon = True
    t.start()

    game_paths = glob(input_dir + "/*.sgf")
    games = [read_sgf(path) for path in game_paths]

    chunks = []
    step = 10
    for i in range(0, len(games), step):
        chunks.append(games[i:min(i + step, len(games))])

    with mp.Pool(initializer=processor_init, initargs=[feat, q]) as pool:
        pool.map(games2dataset, chunks)

    q.join()


def write_input_vectors(features, data, output_dir):
    x = data['x']
    y = data['y']
    fmap = features.split(x)
    fmap['y'] = y
    xshape = list(np.shape(x))
    yshape = list(np.shape(y))

    assert len(xshape) == 4
    assert len(yshape) == 3
    assert xshape[1] == yshape[1] == features.shape[0]
    assert xshape[2] == yshape[2] == features.shape[1]
    assert xshape[3] == features.depth
    assert xshape[0] == yshape[0]

    for k, v in fmap.items():
        with open(join(output_dir, k + ".npy"), 'a+b') as out:
            np.save(out, v)


def writer(q, features, output_dir):
    cnt = 0
    while True:
        chunk = q.get()
        cnt += np.shape(chunk['y'])[0]
        write_input_vectors(features, chunk, output_dir)
        q.task_done()
        print("processed %s moves" % cnt)

if __name__ == '__main__':
    process_sgfs(13, 'data/sgf/sample', 'data/npy/sample')
