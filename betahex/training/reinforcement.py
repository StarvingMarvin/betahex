import tensorflow as tf
import numpy as np
from glob import glob
from random import choice
from tensorflow.contrib import learn

from betahex import Features, Game, Move
from betahex.models import MODEL
from betahex.training.common import make_train_model


def write_sgf(game, path):
    with open(path, 'w') as sgf:
        sgf.write('(;FF[4]GM[11]SZ[13]')
        for move in game.moves:
            sgf.write(';')
            sgf.write(move.__sgf__())
        sgf.write(')')


def generate_batch(features, oponent_data):

    model = make_train_model(
        features,
        policy_filters=MODEL['filters'],
        policy_shape=MODEL['shape'],
        regularization_scale=MODEL['regularization_scale']
    )

    trainee = learn.Estimator(
        model_fn=model,
        model_dir="data/tf/models/reinforcement/%s" % MODEL['name']
    )

    oponent = learn.Estimator(
        model_fn=model,
        model_dir=oponent_data
    )

    games = []

    for i in range(features.board_size * features.board_size):
        norm = Game(features.board_size)
        swapped = Game(features.board_size)
        x = i // features.board_size
        y = i % features.board_size
        norm.play_move(x, y)
        swapped.play_move(x, y)
        swapped.play_swap_pieces()
        games.append(norm)
        games.append(swapped)

    while True:
        running = [g for g in games if not g.winner]

        if not running:
            break

        gen_oponent = [g for g in running
                       if g.next_color == Move.W]

        gen_move = [g for g in running
                    if g.next_color == Move.B]

    # while not game.winner():
    #     board = game.board
    #     if game.next_color == Move.W:
    #         board = board.swap()
    #     X = features.input_map(board)
    #     res = estimator.predict(X, as_iterable=False)
    #     prob = res['probabilities'][0]
    #     pick = np.random.choice(np.arange(len(prob)), p=prob)
    #     print('picked: %s', prob[pick])
    #     print('max: %s', prob[np.max(pick)])
    #     coords = np.unravel_index(pick, features.shape[0:2])
    #     y = coords[1]
    #     x = coords[0] - y // 2
    #
    #     m = Move.make_move(Move.B, 0, x, y)
    #     if game.next_color == Move.W:
    #         m = m.swap()
    #     game.play_move(m.x, m.y)

    return games


def train_batch(games):
    pass


def main(unused_argv):
    features = Features(13, MODEL['features'])

    previous = glob("data/tf/models/reinforcement/%s-*" % MODEL['name'])
    oponent_data = choice(previous)
    oponent_id = oponent_data[oponent_data.rfind('-'):]
    games = generate_batch(features, oponent_data)

    for i, game in enumerate(games):
        write_sgf(game, 'data/sgf/selfplay/b{}-o{}-g{}.sgf'.format(
            len(previous), oponent_id, i)
        )

if __name__ == '__main__':
    tf.app.run()
