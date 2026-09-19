"""Diagnostics for training from a saved feature dataset."""

import pandas as pd


def show_saved_diagnostics(features, model, feature_names, engine):
    estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'feature_importances_'):
        importance = pd.Series(estimator.feature_importances_, index=feature_names)
    elif hasattr(estimator, 'coef_'):
        importance = pd.Series(estimator.coef_[0], index=feature_names)
    else:
        importance = None
    if importance is not None:
        print('\nFeature importance:')
        print(importance.sort_values(key=abs, ascending=False).to_string(float_format=lambda x: f'{x:.3f}'))

    # Recover each team's latest snapshot rating, applying the final completed
    # game's Elo update. Upcoming rows already contain seasonal regression.
    ordered = features.copy()
    ordered['date'] = pd.to_datetime(ordered['date'], utc=True)
    ordered = ordered.sort_values('date', kind='stable')
    ratings = {}
    for _, game in ordered.iterrows():
        names = [str(game['home_team']).strip().lower(), str(game['away_team']).strip().lower()]
        if any(name in {'tbd', 'tba', 'unknown', 'nan', ''} for name in names):
            continue
        home, away = game['home_elo_pre'], game['away_elo_pre']
        if game['completed']:
            change = engine.calculate_elo_change(
                home_elo=home, away_elo=away, home_won=bool(game['home_winner']),
                margin_of_victory=game['home_score'] - game['away_score'],
            )
            home, away = home + change, away - change
        ratings[game['home_id']] = home
        ratings[game['away_id']] = away
    engine.show_power_rankings(ratings, ordered)
