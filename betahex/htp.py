import logging
import re
import numpy as np

from tensorflow.contrib import learn
from betahex import Features, Game, Move
from betahex.models import MODEL
from betahex.training.common import make_train_model

logger = logging.getLogger(__name__)


def canonize(name):
    return ''.join([c if re.match('[a-zA-Z0-9]', c) else '_' for c in name]).lower()


class HtpClient:

    def __init__(self):
        self.game = None
        self.features = None
        self.ai = None

    def handle_command(self, command):
        parts = command.split()
        cmd = parts[0]
        handler = getattr(self, canonize(cmd), None)
        if not callable(handler):
            raise RuntimeError('Unsupported command: ' + cmd)

        return handler(parts)

    def name(self, parts):
        return 'BetaHex'

    def version(self, parts):
        return '0.1'

    def move(self, parts):
        if not self.game:
            raise RuntimeError('Game is not yet initialized')

        if len(parts) != 3:
            raise RuntimeError('Wrong number of arguments. Expected 2 got: %s' % (len(parts) - 1))

        color = parts[1]
        coord = parts[2]

        x = ord(coord[0].upper()) - ord('A')
        y = int(coord[1:])

        self.game.place_move(color, x, y)

    def play(self, parts):
        if not self.game:
            raise RuntimeError('Game is not yet initialized')

        if len(parts) != 3:
            raise RuntimeError('Wrong number of arguments. Expected 2 got: %s' % (len(parts) - 1))

        color = parts[1]
        coord = parts[2]

        x = ord(coord[0].upper()) - ord('A')
        y = int(coord[1:]) - 1

        self.game.play_move(x, y)

    def genmove(self, parts):
        if not self.game:
            raise RuntimeError('Game is not yet initialized')

        board = self.game.board
        if self.game.next_color == Move.W:
            board = board.swap()

        X = self.features.input_map(board)
        res = self.ai.predict(X, as_iterable=False)
        prob = res['probabilities'][0]
        pick = np.random.choice(np.arange(len(prob)), p=prob)
        logger.debug('picked: %s', prob[pick])
        logger.debug('max: %s', prob[np.max(pick)])
        coords = np.unravel_index(pick, self.features.shape[0:2])
        y = coords[1]
        x = coords[0] - y // 2

        m = Move.make_move(Move.B, 0, x, y)
        if self.game.next_color == Move.W:
            m = m.swap()
        self.game.play_move(m.x, m.y)
        return '%c%s' % (ord('a') + m.x, m.y + 1)

    def boardsize(self, parts):
        if self.features:
            raise RuntimeError('Board is already initialized')

        if len(parts) != 3:
            raise RuntimeError('Wrong number of arguments. Expected 2 got: %s' % (len(parts) - 1))
        w = int(parts[1])
        h = int(parts[2])

        if w != 13 or h != 13:
            raise RuntimeError('Only supported size is 13x13')

        self.features = Features(h, MODEL['features'])
        model = make_train_model(
            self.features,
            policy_filters=MODEL['filters'],
            policy_shape=MODEL['shape'])

        self.game = Game(h)
        self.ai = learn.Estimator(
            model_fn=model,
            model_dir="data/tf/models/reinforcement/%s" % MODEL['name']
        )

    def showboard(self, parts):
        if self.game:
            return '\n' + repr(self.game.board)

    def hexgui_analyze_commands(self, parts):
        pass


def ok(message=''):
    print('=', message)
    print('', flush=True)


def error(message):
    print('?', message)
    print('', flush=True)


def main():

    logging.basicConfig(filename='htp.log', level=logging.DEBUG)

    htp = HtpClient()

    while True:
        cmd = input().strip()
        logger.debug("received: %s", cmd)
        if cmd == 'quit':
            ok()
            break

        try:
            res = htp.handle_command(cmd)
            ok(res or '')
            logger.debug("responded: %s", res)
        except Exception as e:
            logger.exception(str(e))
            error(str(e))
            break


if __name__ == '__main__':
    main()
