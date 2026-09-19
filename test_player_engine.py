import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np
import pandas as pd

from player_engine import PlayerHistory, COUNTS, STATS, normalize_boxes
from awards_predictor import season_stats, award_dataset, candidates, walk_forward, BASE


def boxes():
    rows = []
    for index in range(3):
        for player, team in [(101,1),(102,1),(201,2),(202,2)]:
            row = {c: 1. for c in COUNTS}
            row.update(game_id=index+1, season=2025, season_type=2,
                       date=pd.Timestamp('2025-01-01',tz='UTC')+pd.Timedelta(days=index*2),
                       player_id=player, player_name=f'Player {player}', team_id=team,
                       home_id=1, away_id=2, opponent_id=3-team, home_score=100,away_score=90,
                       won=int(team==1), minutes=30., points=20.+index, rebounds=5.,
                       field_goals_made=8.,field_goals_attempted=16.,free_throws_made=4.,free_throws_attempted=5.,
                       game_score=15., player_elo=1500.+index)
            rows.append(row)
    return pd.DataFrame(rows)


class PlayerTests(unittest.TestCase):
    def test_current_and_future_box_scores_do_not_leak(self):
        original = boxes()
        altered = original.copy()
        altered.loc[altered.game_id >= 2, ['points','game_score']] = 999.
        left, right = PlayerHistory(original,'nba'), PlayerHistory(altered,'nba')
        timestamp = original.loc[original.game_id==2,'date'].iloc[0]
        left.before(timestamp); right.before(timestamp)
        self.assertEqual(left.features(1,2025),right.features(1,2025))
        self.assertEqual(left.ratings,right.ratings)
        right.before(timestamp+pd.Timedelta(seconds=1))
        self.assertNotEqual(left.features(1,2025),right.features(1,2025))

    def test_ratings_update_once_and_survive_trade(self):
        data=boxes()
        data.loc[(data.game_id==2)&(data.player_id==101), ['team_id','opponent_id','won']] = [2,1,0]
        history=PlayerHistory(data,'nba')
        time=data.loc[data.game_id==2,'date'].iloc[0]+pd.Timedelta(seconds=1)
        history.before(time)
        self.assertEqual(history.last_team[101],2)
        self.assertNotEqual(history.ratings[101],1500)
        rating=history.ratings.copy()
        history.before(time)
        self.assertEqual(rating,history.ratings)

    def test_current_rosters_used_only_for_forecasts(self):
        roster=pd.DataFrame([dict(season=2025,team_id=1,athlete_id='201')])
        history=PlayerHistory(boxes(),'nba',roster)
        history.before(pd.Timestamp('2025-01-02',tz='UTC'))
        historical=history.features(1,2025,forecast=False)
        future=history.features(1,2025,forecast=True)
        self.assertNotEqual(historical['player_elo_adjustment'],future['player_elo_adjustment'])

    def test_shooting_uses_attempt_weighted_percentage(self):
        data=boxes().query('game_id == 1').copy()
        data.loc[data.player_id==101,['field_goals_made','field_goals_attempted']]=[1,1]
        data.loc[data.player_id==102,['field_goals_made','field_goals_attempted']]=[1,9]
        history=PlayerHistory(data,'wnba'); history.before(pd.Timestamp('2025-01-02',tz='UTC'))
        self.assertAlmostEqual(history.features(1,2025)['player_fg_pct'],.2)

    def test_dnp_unplayed_and_duplicate_rows_excluded(self):
        data=boxes().rename(columns={'player_id':'athlete_id','player_name':'athlete_display_name'})
        data.loc[data.athlete_id==102,'minutes']=0
        data=pd.concat([data,data.iloc[[0]]],ignore_index=True)
        games=boxes().drop_duplicates('game_id').copy()
        games['status_type_completed'] = games.game_id < 3
        with __import__('warnings').catch_warnings():
            normalized=normalize_boxes(data,games)
        self.assertEqual(normalized.game_id.nunique(),2)
        self.assertNotIn(102,normalized.player_id.values)
        self.assertEqual(len(normalized),6)


class AwardTests(unittest.TestCase):
    def test_regular_season_only_and_prior_year_improvement(self):
        old=boxes().copy(); old['season']=2024; old['points']=10
        new=boxes().copy(); new['season']=2025; new['points']=20
        playoff=new.copy(); playoff['season_type']=3; playoff['points']=999
        stats=season_stats(pd.concat([old,new,playoff],ignore_index=True))
        row=stats[(stats.season==2025)&(stats.player_id==101)].iloc[0]
        self.assertEqual(row.points,20)
        self.assertEqual(row.points_gain,10)
        self.assertTrue(row.has_previous_season)
        self.assertFalse(row.rookie_inferred)

    def test_co_winners_and_past_awards_only(self):
        old=boxes().copy(); old['season']=2024
        new=boxes().copy(); new['season']=2025
        stats=season_stats(pd.concat([old,new],ignore_index=True))
        history=pd.DataFrame([dict(award='DPOY',season=2024,player_name='Player 101'),
                              dict(award='DPOY',season=2025,player_name='Player 101'),
                              dict(award='DPOY',season=2025,player_name='Player 201')])
        labelled=award_dataset(stats,history,'DPOY')
        self.assertEqual(labelled.query('season == 2025').winner.sum(),2)
        self.assertEqual(labelled.query('season == 2024').previous_awards.sum(),0)
        self.assertEqual(labelled.query('season == 2025 and player_id == 101').previous_awards.iloc[0],1)

    def test_unmatched_winner_is_not_silently_a_negative(self):
        stats=season_stats(boxes())
        history=pd.DataFrame([dict(award='MVP',season=2025,player_name='Missing Person')])
        with self.assertRaisesRegex(ValueError,'not matched'):
            award_dataset(stats,history,'MVP')

    def test_award_backtests_train_strictly_earlier_seasons(self):
        data=[]
        for year in range(2019,2025):
            for player in range(4):
                row={c:float(player+1) for c in BASE}
                row.update(season=year,player_name=str(player),games=30,winner=int(player==3),rookie_inferred=False,has_previous_season=True)
                data.append(row)
        result=walk_forward(pd.DataFrame(data),'MVP')
        self.assertEqual(len(result),3)
        self.assertTrue(all(r['train_through'] < r['season'] for r in result))


if __name__ == '__main__':
    unittest.main()
