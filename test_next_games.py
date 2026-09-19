import contextlib
import io
import unittest

import pandas as pd

from console_output import next_games_by_team, show_predictions


def predictions(rows):
    result = pd.DataFrame(rows, columns=['date', 'away_team', 'home_team'])
    result['predicted_winner'] = result['home_team']
    result['confidence'] = 0.67891
    return result


class NextGameTests(unittest.TestCase):
    def test_all_thirty_teams_without_five_game_limit(self):
        data = predictions([('2099-10-01', f'Team{i}', f'Team{i+1}') for i in range(0, 30, 2)])
        result = next_games_by_team(data, now='2099-09-01')
        self.assertEqual(len(result), 15)
        self.assertEqual(len(set(result.away_team) | set(result.home_team)), 30)

    def test_shared_matchup_once_and_every_teams_actual_next_game(self):
        data = predictions([
            ('2099-10-03', 'A', 'B'),
            ('2099-10-02', 'A', 'C'),
            ('2099-10-01', 'A', 'B'),
            ('2099-10-01', 'A', 'B'),
        ])
        result = next_games_by_team(data, now='2099-09-01')
        self.assertEqual(list(zip(result.away_team, result.home_team)), [('A', 'B'), ('A', 'C')])

    def test_stale_and_invalid_dates_are_excluded(self):
        data = predictions([('2020-01-01', 'A', 'B'), ('invalid', 'C', 'D'), ('2099-10-01', 'A', 'C')])
        result = next_games_by_team(data, now='2099-09-01')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].home_team, 'C')

    def test_output_format_and_full_verbose_schedule(self):
        data = predictions([('2099-10-01', 'A', 'B'), ('2099-10-02', 'A', 'B')])
        before = data.copy(deep=True)
        compact, verbose = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(compact):
            show_predictions(data)
        with contextlib.redirect_stdout(verbose):
            show_predictions(data, verbose=True)
        self.assertIn("2 teams, 1 matchups", compact.getvalue())
        self.assertIn('67.9%', compact.getvalue())
        self.assertNotIn('2099-10-02', compact.getvalue())
        self.assertIn('2099-10-02', verbose.getvalue())
        pd.testing.assert_frame_equal(data, before)

    def test_empty_schedule_is_clear(self):
        data = predictions([])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            show_predictions(data)
        self.assertIn('No upcoming games available', output.getvalue())

    def test_unknown_opponents_do_not_count_as_teams(self):
        data = predictions([('2099-10-01', 'A', ' TBD '), ('2099-10-01', None, 'B'), ('2099-10-02', 'A', 'B')])
        result = next_games_by_team(data, now='2099-09-01')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].home_team, 'B')


if __name__ == '__main__':
    unittest.main()
