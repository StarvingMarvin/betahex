import os
import tensorflow as tf

from betahex.utils.sgf import read_sgf
from betahex.models.policy import Policy

p = Policy(13)
moves = read_sgf('data/sgfs/train/1000211.sgf')
boards = p.moves2boards(moves)
f = p.board2features(boards[0][10])

model = p.model()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(model, feed_dict={p.x: [p.board2features(b) for b in boards[0]]})