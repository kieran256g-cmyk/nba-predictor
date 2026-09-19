"""Point-in-time player box-score features and custom individual Elo ratings."""
from collections import defaultdict, deque
from pathlib import Path
import importlib
import json
import warnings

import numpy as np
import pandas as pd

STATS = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
         'fg_pct', 'three_pct', 'ft_pct', 'minutes']
COUNTS = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'minutes',
          'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
          'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
          'offensive_rebounds', 'defensive_rebounds', 'fouls']
RATIOS = {'fg_pct': ('field_goals_made', 'field_goals_attempted'),
          'three_pct': ('three_point_field_goals_made', 'three_point_field_goals_attempted'),
          'ft_pct': ('free_throws_made', 'free_throws_attempted')}
FEATURES = ([f'player_{s}_diff' for s in STATS]
            + [f'player_{s}_diff_{w}' for w in [5, 10] for s in STATS]
            + ['player_elo_adjustment_diff', 'player_history_coverage_diff'])


def _loader(league, kind):
    return getattr(importlib.import_module(f'sportsdataverse.{league}'), f'load_{league}_{kind}')


def load_release(league, kind, seasons, offline=False):
    """Refresh each requested release; offline use is always explicit."""
    directory = Path('data/cache')
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in sorted(set(map(int, seasons))):
        path = directory / f'{league}_{kind}_{season}.parquet'
        if offline:
            if not path.exists():
                raise FileNotFoundError(f'Missing cache {path}; run once online first.')
            frame = pd.read_parquet(path)
        else:
            frame = _loader(league, kind)([season], return_as_pandas=True)
            # Write only successful responses. Connection errors never silently
            # substitute older data into a supposedly refreshed run.
            frame.to_parquet(path, index=False)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rosters(league, seasons, offline=False):
    supported = 2025 if league == 'nba' else 2024
    return load_release(league, 'rosters', [s for s in seasons if s >= supported], offline)


def normalize_boxes(raw, games):
    """Keep played appearances with trustworthy completed-game labels."""
    required = set(COUNTS + ['game_id', 'team_id', 'athlete_id', 'athlete_display_name'])
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Player box scores missing columns: {sorted(missing)}')
    boxes = raw[list(required)].copy()
    for column in ['game_id', 'team_id', 'athlete_id'] + COUNTS:
        boxes[column] = pd.to_numeric(boxes[column], errors='coerce')
    boxes = boxes[boxes['minutes'] > 0].dropna(subset=['game_id', 'team_id', 'athlete_id'])
    if boxes[COUNTS].isna().any().any():
        raise ValueError('Played player box scores contain missing statistics; refusing to treat missing data as zero.')
    for column in ['game_id', 'team_id', 'athlete_id']:
        boxes[column] = boxes[column].astype('int64')
    schedule = games[games['status_type_completed'].fillna(False)].copy()
    schedule['date'] = pd.to_datetime(schedule['date'], utc=True)
    schedule = schedule[schedule['date'] < pd.Timestamp.now(tz='UTC')]
    schedule = schedule.drop_duplicates('game_id')
    columns = ['game_id', 'date', 'season', 'season_type', 'home_id', 'away_id', 'home_score', 'away_score']
    boxes = boxes.merge(schedule[columns], on='game_id', how='inner', validate='many_to_one')
    boxes = boxes[(boxes.team_id == boxes.home_id) | (boxes.team_id == boxes.away_id)]
    boxes['won'] = np.where(boxes.team_id == boxes.home_id, boxes.home_score > boxes.away_score,
                            boxes.away_score > boxes.home_score).astype(int)
    boxes['opponent_id'] = np.where(boxes.team_id == boxes.home_id, boxes.away_id, boxes.home_id)
    boxes = boxes.rename(columns={'athlete_id': 'player_id', 'athlete_display_name': 'player_name'})
    boxes = boxes.drop_duplicates(['game_id', 'team_id', 'player_id']).sort_values(['date', 'game_id', 'player_id'])
    # Partial boxes should not update only one side's player ratings.
    valid_games = boxes.groupby('game_id').team_id.nunique()
    boxes = boxes[boxes.game_id.isin(valid_games[valid_games == 2].index)].copy()
    coverage = boxes.game_id.nunique() / max(1, len(schedule))
    if coverage < .95:
        warnings.warn(f'Player boxes cover {coverage:.1%} of completed games. Missing games have no player update.')
    boxes['game_score'] = (boxes.points + .4*boxes.field_goals_made - .7*boxes.field_goals_attempted
                           - .4*(boxes.free_throws_attempted-boxes.free_throws_made)
                           + .7*boxes.offensive_rebounds + .3*boxes.defensive_rebounds
                           + boxes.steals + .7*boxes.assists + .7*boxes.blocks - .4*boxes.fouls - boxes.turnovers)
    return boxes.reset_index(drop=True)


class PlayerHistory:
    """Advance box-score state only after a game's timestamp, never before it."""
    def __init__(self, boxes, league, rosters=None):
        self.league = league
        self.boxes = boxes.copy()
        self.groups = iter(boxes.groupby(['date', 'game_id'], sort=True)) if not boxes.empty else iter([])
        self.pending = next(self.groups, None)
        self.history = defaultdict(lambda: deque(maxlen=10))
        self.team_games = defaultdict(lambda: deque(maxlen=5))
        self.ratings, self.last_season, self.last_team, self.names, self.last_date = {}, {}, {}, {}, {}
        self.player_log = []
        self.rosters = pd.DataFrame() if rosters is None else rosters.copy()
        self.forecast_season = int(self.rosters.season.max()) if not self.rosters.empty else None
        self._projection_cache = {}

    def before(self, date):
        date = pd.to_datetime(date, utc=True)
        while self.pending is not None and self.pending[0][0] < date:
            _, frame = self.pending
            self._update(frame)
            self.pending = next(self.groups, None)

    def _update(self, frame):
        self._projection_cache.clear()
        season = int(frame.iloc[0].season)
        for row in frame.itertuples():
            pid = row.player_id
            if pid not in self.ratings:
                self.ratings[pid] = 1500.
            elif self.last_season[pid] != season:
                self.ratings[pid] = 1500 + .75*(self.ratings[pid]-1500)
            self.last_season[pid] = season
        team_strength = {}
        for tid, team in frame.groupby('team_id'):
            team_strength[tid] = float(np.average([self.ratings[p] for p in team.player_id], weights=team.minutes))
        # Calculate all deltas against the pre-game rating state.
        for row in frame.itertuples():
            pid, old = row.player_id, self.ratings[row.player_id]
            advantage = 65 if row.team_id == row.home_id else -65
            expected = 1/(1+10**((team_strength[row.opponent_id]-team_strength[row.team_id]-advantage)/400))
            participation = min(row.minutes / (48 if self.league == 'nba' else 40), 1.)
            outcome_delta = 20 * participation * (row.won-expected)
            # Box-score contribution is a bounded heuristic, not causal impact.
            performance_target = 1500 + np.clip((row.game_score/max(row.minutes, 1)*36 - 12)*12, -200, 200)
            performance_delta = .05 * participation * (performance_target-old)
            self.ratings[pid] = float(old + outcome_delta + performance_delta)
            record = row._asdict()
            record['player_elo'] = self.ratings[pid]
            record['player_elo_adjustment'] = self.ratings[pid]-1500
            record['elo_change'] = self.ratings[pid]-old
            self.player_log.append(record)
            self.history[pid].append(record)
            self.last_team[pid], self.names[pid], self.last_date[pid] = row.team_id, row.player_name, row.date
        for tid, team in frame.groupby('team_id'):
            self.team_games[tid].append(set(team.player_id))

    def features(self, team_id, season, forecast=False):
        key = (team_id, season, forecast)
        if key in self._projection_cache:
            return self._projection_cache[key].copy()
        members = set().union(*self.team_games[team_id]) if self.team_games[team_id] else set()
        members = {p for p in members if self.last_team.get(p) == team_id}
        if forecast and self.forecast_season == season:
            roster = self.rosters[(self.rosters.season == season) & (self.rosters.team_id == team_id)]
            if not roster.empty:
                members = set(pd.to_numeric(roster.athlete_id, errors='coerce').dropna().astype(int))
        known = [p for p in members if self.history[p]]
        result = {f'player_{s}': 0. for s in STATS}
        result.update({f'player_{s}_{w}': 0. for w in [5,10] for s in STATS})
        result['player_elo_adjustment'] = 0.
        result['player_history_coverage'] = len(known)/max(len(members),1)
        minutes5 = []
        for window in [5, 10]:
            total = {s: 0. for s in COUNTS}
            for pid in known:
                rows = list(self.history[pid])[-window:]
                averages = np.asarray([[r[s] for s in COUNTS] for r in rows], dtype=float).mean(axis=0)
                for stat, value in zip(COUNTS, averages):
                    total[stat] += value
                if window == 5:
                    minutes5.append(averages[COUNTS.index('minutes')])
            scale = (240 if self.league == 'nba' else 200)/max(total['minutes'], 1) if known else 0
            for stat in STATS:
                if stat in RATIOS:
                    made, attempts = RATIOS[stat]
                    value = total[made]/max(total[attempts],1)
                else:
                    value = total[stat]*scale
                result[f'player_{stat}_{window}'] = float(value)
                if window == 5:
                    result[f'player_{stat}'] = float(value)
        if known and sum(minutes5) > 0:
            ratings = [1500 + (self.ratings[p]-1500)*(.75 if self.last_season[p] != season else 1) for p in known]
            result['player_elo_adjustment'] = float(np.average(ratings, weights=minutes5)-1500)
        self._projection_cache[key] = result.copy()
        return result

    def export(self):
        self.before(pd.Timestamp.now(tz='UTC'))
        records = pd.DataFrame(self.player_log)
        records.to_csv('player_game_stats.csv', index=False)
        current = pd.DataFrame([dict(player_id=p, player_name=self.names[p], team_id=self.last_team[p],
                                     player_elo=self.ratings[p], player_elo_adjustment=self.ratings[p]-1500,
                                     last_game=self.last_date[p]) for p in self.ratings])
        current = current.sort_values('player_elo', ascending=False) if not current.empty else current
        current['last_played_team_id'] = current['team_id']
        current['active_roster'] = False
        if not self.rosters.empty:
            roster = self.rosters[self.rosters.season == self.forecast_season].copy()
            roster['player_id'] = pd.to_numeric(roster.athlete_id, errors='coerce')
            # Only unambiguous current roster entries can override team labels.
            roster = roster.dropna(subset=['player_id']).drop_duplicates(['player_id','team_id'])
            roster = roster[~roster.player_id.duplicated(keep=False)].set_index('player_id')
            current['active_roster'] = current.player_id.isin(roster.index)
            current['team_id'] = current.player_id.map(roster.team_id).fillna(current.team_id)
        current.to_csv('player_ratings.csv', index=False)
        print(f'Updated {len(current)} player ratings; full stats in player_game_stats.csv and player_ratings.csv.')
        active = current[current.active_roster]
        if not active.empty:
            print('\nPLAYER ELO - top 10 on the latest roster snapshot (custom rating):')
            print(active[['player_name','team_id','player_elo','player_elo_adjustment']].head(10).to_string(index=False, float_format=lambda x: f'{x:.1f}'))
        return records, current


def add_player_features(feature_row, home, away):
    for stat in STATS:
        feature_row[f'player_{stat}_diff'] = home[f'player_{stat}']-away[f'player_{stat}']
        for window in [5,10]:
            feature_row[f'player_{stat}_diff_{window}'] = home[f'player_{stat}_{window}']-away[f'player_{stat}_{window}']
    for stat in ['player_elo_adjustment', 'player_history_coverage']:
        feature_row[f'{stat}_diff'] = home[stat]-away[stat]
    return feature_row
