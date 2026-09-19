import contextlib
import io
import unittest

import pandas as pd

from console_output import show_predictions
from nba_predictor import FEATURE_COLUMNS, compute_features, remove_exhibition_games


def fixtures():
    return pd.DataFrame([
        dict(game_id=i, season=2026, season_type=2,
             date=pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(days=i),
             status_type_completed=True, home_id=2, away_id=13,
             home_display_name='Boston Celtics', away_display_name='Los Angeles Lakers',
             home_score=110, away_score=100, home_winner=True, away_winner=False,
             neutral_site=False)
        for i in range(3)
    ])


class PredictorTests(unittest.TestCase):
    def test_excludes_exhibition_preseason_and_unknown_teams(self):
        games = pd.concat([fixtures()] * 2, ignore_index=True)
        games.loc[0, 'home_id'] = 130579
        games.loc[1, 'away_id'] = -1
        games.loc[2, 'season_type'] = 1
        games.loc[4, 'season_type'] = 3
        games.loc[5, 'season_type'] = 5
        self.assertEqual(remove_exhibition_games(games).index.tolist(), [3, 4, 5])

    def test_current_outcome_does_not_leak_into_its_features(self):
        games = fixtures()
        original, _ = compute_features(games)
        games.loc[1, ['home_score', 'away_score', 'home_winner', 'away_winner']] = [90, 120, False, True]
        changed, _ = compute_features(games)
        pd.testing.assert_frame_equal(original.loc[:1, FEATURE_COLUMNS], changed.loc[:1, FEATURE_COLUMNS])
        self.assertNotEqual(original.loc[2, 'elo_diff'], changed.loc[2, 'elo_diff'])

    def test_unplayed_scores_do_not_update_ratings(self):
        games = fixtures()
        games.loc[1:, 'status_type_completed'] = False
        original, ratings = compute_features(games)
        games.loc[1:, 'home_score'] = 999
        changed, changed_ratings = compute_features(games)
        pd.testing.assert_frame_equal(original[FEATURE_COLUMNS], changed[FEATURE_COLUMNS])
        self.assertEqual(ratings, changed_ratings)



if __name__ == '__main__':
    unittest.main()
