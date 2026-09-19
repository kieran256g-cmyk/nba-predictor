import unittest
import numpy as np
from confidence_check import confidence_table


class ConfidenceTests(unittest.TestCase):
    def test_seventy_percent_picks_win_seventy_percent(self):
        # Five home picks and five away picks, seven correct in total.
        table = confidence_table([.7] * 5 + [.3] * 5, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        row = table.iloc[2]
        self.assertEqual(row.games, 10)
        self.assertEqual(row.wins, 7)
        self.assertAlmostEqual(row.predicted, .7)
        self.assertAlmostEqual(row.actual, .7)
        self.assertAlmostEqual(row.gap_pp, 0)

    def test_boundaries_and_certain_predictions(self):
        table = confidence_table([.5, .6, .7, .8, .9, 1., 0., .4], [1] * 8)
        self.assertEqual(table.games.tolist(), [1, 2, 1, 1, 3])
        self.assertEqual(table.games.sum(), 8)

    def test_overconfidence_gap_and_empty_bins(self):
        table = confidence_table([.75, .75], [1, 0])
        self.assertAlmostEqual(table.iloc[2].gap_pp, -25)
        self.assertTrue(np.isnan(table.iloc[0].actual))
        self.assertEqual(confidence_table([], []).games.sum(), 0)

    def test_invalid_input(self):
        for probabilities, outcomes in [([.7], []), ([1.1], [1]), ([np.nan], [0]), ([.7], [2])]:
            with self.assertRaises(ValueError):
                confidence_table(probabilities, outcomes)


if __name__ == '__main__':
    unittest.main()
