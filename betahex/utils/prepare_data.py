import multiprocessing as mp
from glob import glob
from threading import Thread
import random

import tensorflow as tf
from os.path import join

from betahex.features import Features
from betahex.game import Board, Move
from betahex.utils.sgf import read_sgf


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
        kwargs={'q': q, 'output_dir': output_dir},
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


def writer(q, output_dir):
    train_filename = join(output_dir, 'train.tfrecords')
    eval_filename = join(output_dir, 'eval.tfrecords')
    test_filename = join(output_dir, 'test.tfrecords')

    cnt = 0
    cnt_train = 0
    cnt_eval = 0
    cnt_test = 0
    opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(train_filename, opt) as train:
        with tf.python_io.TFRecordWriter(eval_filename, opt) as eval:
            with tf.python_io.TFRecordWriter(test_filename, opt) as test:
                while True:
                    chunk = q.get(timeout=60)
                    cnt += len(chunk)
                    for example in chunk:
                        ser = example.SerializeToString()
                        switch = random.random()
                        if switch < 0.7:
                            writer = train
                            cnt_train += 1
                        elif switch < 0.85:
                            writer = eval
                            cnt_eval += 1
                        else:
                            writer = test
                            cnt_test += 1
                        writer.write(ser)
                    q.task_done()
                    print("inc", len(chunk))
                    print("processed %s moves" % cnt)
                    print("train, eval, test: %s, %s, %s" % (cnt_train, cnt_eval, cnt_test))


if __name__ == '__main__':
    process_sgfs(13, 'data/sgf/sample', 'data/tf/features')
