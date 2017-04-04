import tensorflow as tf
from os.path import join, dirname

from betahex.game import moves2boards
from betahex.utils.sgf import read_sgf
from betahex.models.policy import Policy

p = Policy(13)
moves = read_sgf(join(dirname(__file__), '../test/data/1000211.sgf'))
boards = moves2boards(13, moves)
f = p.features(boards[0][10])

model = p.model()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(model, feed_dict={p.x: [p.features(b) for b in boards[0]]})
