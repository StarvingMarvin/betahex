import multiprocessing as mp
from glob import glob
from threading import Thread

import numpy as np
import tensorflow as tf
from os.path import join

from betahex.features import Features
from betahex.game import Board, Move
from betahex.utils.sgf import read_sgf


# def feature_pairs(f, board, move):
#     normal = f.input_map(board)
#     normal['y'] = f.one_hot_move(move)
#
#     rot = f.input_map(board.rotate())
#     rot['y'] = f.one_hot_move(move.rotate(board.shape()[0]))
#
#     return ret


def moves2dataset(f, moves):
    prev = Board.make_of_size(f.board_size)

    ret = []

    for m in moves:
        if m.special == Move.SWAP_PIECES:
            prev = prev.swap()
            continue
        cur = prev
        y = m
        if m.color == Move.W:
            cur = prev.swap()
            y = m.swap()

        ret.append(f.input_example(cur, y))
        ret.append(f.input_example(cur.rotate(), y.rotate(f.board_size)))

        prev = prev.place_move(m)

    return ret


def games2dataset(games):
    data = []
    for moves in games:
        data.extend(moves2dataset(games2dataset.f, moves))

    games2dataset.q.put(data)


def processor_init(f, q):
    games2dataset.f = f
    games2dataset.q = q


def process_sgfs(board_size, input_dir, output_dir):

    q = mp.JoinableQueue()
    feat = Features(board_size)

    t = Thread(
        target=writer,
        kwargs={'q': q, 'features': feat, 'output_dir': output_dir},
        daemon=True
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


# def write_input_vectors(features, data, output_file):
#     x = data['x']
#     y = data['y']
#     fmap = features.split(x)
#     xshape = list(np.shape(x))
#     yshape = list(np.shape(y))
#
#     assert len(xshape) == 4
#     assert len(yshape) == 3
#     assert xshape[1] == yshape[1] == features.shape[0]
#     assert xshape[2] == yshape[2] == features.shape[1]
#     assert xshape[3] == features.shape[2]
#     assert xshape[0] == yshape[0]
#
#     for k, v in fmap.items():
#         if k not in output_file.root.x:
#             print("creating array for", k)ther account
#             output_file.create_earray(
#                 output_file.root.x, k,
#                 atom=tables.Int8Atom(),
#                 shape=(0,) + np.shape(v)[1:]
#             )
#
#         fa = output_file.get_node(output_file.root.x, k)
#         fa.append(v)
#     ya = output_file.get_node(output_file.root.y)
#     ya.append(y)


def writer(q, features, output_dir):
    filename = join(output_dir, 'main.tfrecords')

    # f = tables.open_file(filename, mode='w')
    # f.create_group(f.root, 'x')
    # f.create_earray(f.root, 'y', atom=tables.Int8Atom(), shape=(0,)+features.shape[0:2])
    cnt = 0
    with tf.python_io.TFRecordWriter(filename) as writer:
        while True:
            chunk = q.get()
            cnt += len(chunk)
            for example in chunk:
                writer.write(example.SerializeToString())
            q.task_done()
            print("inc", len(chunk))
            print("processed %s moves" % cnt)

if __name__ == '__main__':
    process_sgfs(13, 'data/sgf/main', 'data/tf/features')
