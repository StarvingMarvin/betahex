import tensorflow as tf
import sys

COMMAND_HANDLERS = {
    'name': lambda _: 'BetaHex',
    'version': lambda _: '0.1',
    'move': lambda move: '',
    'play': lambda move: '',
    'genmove': lambda color: 'b7',
    'hexgui-analyze_commands': lambda _: '',
    'boardsize': lambda _: '',
    'showboard': lambda _: '',
}


def main():

    log = open('htp.log', 'w')

    while True:
        cmd = input().strip()
        log.write("received: " + cmd + '\n')
        log.flush()
        if cmd == 'quit':
            break

        parts = cmd.split()
        cmd_start = parts[0]

        try:
            res = COMMAND_HANDLERS[cmd_start](parts)
            print('=', res)
            print('', flush=True)
            log.write("responded: " + res + '\n')
            log.flush()
        except Exception as e:
            print('?', str(e), flush=True)
            print('', flush=True)
            break

    log.close()


if __name__ == '__main__':
    main()
