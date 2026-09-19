"""Compare picked-team confidence with observed wins on validation games."""

import numpy as np
import pandas as pd


def confidence_table(home_probabilities, home_winners):
    probabilities = np.asarray(home_probabilities, dtype=float)
    outcomes = np.asarray(home_winners)
    if probabilities.ndim != 1 or outcomes.ndim != 1 or len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must be matching one-dimensional arrays.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Outcomes must be zero or one.")
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (probabilities >= 0.5) == outcomes
    rows = []
    for low, high in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
        # Round only for stable boundary assignment (e.g. 1 - 0.4).
        percentage = np.round(confidence * 100, 10)
        mask = (percentage >= low) & ((percentage < high) if high < 100 else (percentage <= high))
        count = int(mask.sum())
        expected = float(confidence[mask].mean()) if count else np.nan
        observed = float(correct[mask].mean()) if count else np.nan
        rows.append({"band": f"{low}-<{high}%" if high < 100 else "90-100%",
                     "games": count, "wins": int(correct[mask].sum()),
                     "predicted": expected, "actual": observed,
                     "gap_pp": (observed - expected) * 100})
    return pd.DataFrame(rows)


def show_confidence_check(home_probabilities, home_winners):
    table = confidence_table(home_probabilities, home_winners)
    print("\nCONFIDENCE CHECK (held-out validation games)")
    display = table.copy()
    for column in ['predicted', 'actual']:
        display[column] = display[column].map(lambda x: '-' if pd.isna(x) else f'{x:.1%}')
    display['gap_pp'] = display['gap_pp'].map(lambda x: '-' if pd.isna(x) else f'{x:+.1f}')
    display['sample'] = table['games'].map(lambda n: 'No games' if n == 0 else ('Small sample' if n < 30 else ''))
    display.columns = ['Confidence', 'Games', 'Wins', 'Avg predicted', 'Actual win rate', 'Gap (pp)', 'Sample']
    print(display.to_string(index=False))
    print('Negative gap = overconfident; positive gap = underconfident.')
    print('Checks both home and away picks before refitting. These games also select the model; this is not an independent final test.')
    return table
