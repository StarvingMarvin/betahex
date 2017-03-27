import urllib
import time
import glob
import re
from bs4 import BeautifulSoup

# Post fact copy-paste from interactive console. All important bits should be
# here, but are probably not connected properly


def download_tournaments(tournaments, outdir='tournaments'):
    tournament_template = 'http://littlegolem.net/jsp/tournament/tournament.jsp?trnid={}'

    for t in tournaments:
        urllib.request.urlretrieve(tournament_template.format(t), '{}/{}.html'.format(outdir, t))
        time.sleep(1)


def parse_player(td):
    name = td.contents[0].string.strip()
    href = td.find('a').get('href')
    id = href[href.rfind('=') + 1:]
    return name, id


def get_players_from_tournaments(file_paths):
    res = set()
    for f in file_paths:
        with open(f) as t:
            page = BeautifulSoup(t, "lxml")
            table = page.find('table')
            res.update({parse_player(row.find_all('td')[1]) for row in table.find_all('tr')[2:]})
    return res


def download_players(players, outdir='players'):
    player_template = 'http://littlegolem.net/jsp/info/player_game_list.jsp?gtid=hex&plid={}'
    player_file_template = '{}/{}.html'

    for p in players:
        urllib.request.urlretrieve(player_template.format(p[1]), player_file_template.format(outdir, p[1]))
        time.sleep(1)


def download_top_players(page_count, outdir='tournaments'):
    top_url_template = 'http://littlegolem.net/jsp/info/player_list.jsp?gtvar=hex_DEFAULT&filter=&countryid=&page={}'
    top_file_template = '{}/{}.html'
    for i in range(page_count):
        urllib.request.urlretrieve(top_url_template.format(i), top_file_template.format(outdir, i + 1))
        time.sleep(1)
    

def parse_player_from_top(a):
    href = a.get('href')
    id = href[href.rfind('=') + 1:]
    name = a.contents[0]
    return name, id


def get_players_from_top(files):
    res = set()
    for p in files:
        with open(p) as top:
            page = BeautifulSoup(top, 'lxml')
            portlet = page.find('div', class_='portlet-body')
            res.update({parse_player_from_top(a) for a in portlet.find_all('a')})
    return res
        

def filter_rows(rows, names):
    res = set()
    for row in rows:
        cols = [c for c in row.contents if c.name]
        try:
            if len(cols) < 6:
                print(cols)
            elif cols[1].contents[0] in names and int(cols[4].contents[0]) > 20 and cols[5].contents[0].strip():
                res.add(cols[0].find('a').contents[0])
        except IndexError as e:
            print(cols)
    return res


def collect_games_from_players():
    games = set()
    for p in glob.glob('players/*'):
        with open(p) as f:
            page = BeautifulSoup(f, 'lxml')
            r = page.find_all('table')[1].find_all('tr')[1:]
            games.update(filter_rows(r))
    return games
        

def download_games(game_ids, outdir='games'):
    game_url_template = 'http://littlegolem.net/jsp/game/game.jsp?gid={}'
    game_file_template = '{}/{}.html'

    for gid in game_ids:
        urllib.request.urlretrieve(game_url_template.format(gid), game_file_template.format(outdir, gid))
        time.sleep(1)


def page2sgf(page, sgf):
    ml = page.find(string=re.compile("Move List"))
    moves = ml.parent.parent.parent.find(class_='portlet-body').find_all('b')
    sgf.write('(;FF[4]GM[11]SZ[13]')
    color = 'B'

    def flip_color():
        return 'B' if color == 'W' else 'W'

    def write_move(coord):
        sgf.write(';')
        sgf.write(color)
        sgf.write('[')
        sgf.write('swap-pieces' if coord == 'swap' else coord)
        sgf.write(']')

    for move in moves:
        coord = move.string.split('.')[1]
        write_move(coord)
        color = flip_color()
    sgf.write(')')


def scrape_lg(datadir='data'):
    tournaments = ['hex.ch.{}.{}.{}'.format(i, j, k) for i, in range(10, 38) for j, k in [(1, 1), (1, 2), (2, 2)]]
    download_tournaments(tournaments, datadir + '/tournaments')
    ch_players = get_players_from_tournaments(glob.glob(datadir + '/tournaments/hex.ch.*.html'))
    top_players = get_players_from_top(glob.glob('tournaments/top-*.html'))
    stars = [p for p in ch_players if p[0].find('★') > -1]
    new_players = top_players - ch_players
    new_players -= {(p[0][:-2], p[1]) for p in stars}

    players = ch_players | new_players
    names = {p[0] for p in players}
    games = set()
    game_ids = [g.strip(' #') for g in games]

    for gid in game_ids:
        with open('games/{}.html'.format(gid)) as html:
            with open('games/{}.sgf'.format(gid), 'w') as sgf:
                page = BeautifulSoup(html, 'lxml')
                page2sgf(page, sgf)
