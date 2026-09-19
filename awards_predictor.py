"""Historical-winner models for MVP, ROY, DPOY and MIP (ranking scores, not odds)."""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AWARDS = ['MVP', 'ROY', 'DPOY', 'MIP']
BASE = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'minutes',
        'true_shooting', 'team_win_pct', 'games_share', 'player_elo', 'game_score',
        'points_gain', 'rebounds_gain', 'assists_gain', 'minutes_gain', 'game_score_gain',
        'previous_awards']


def name_key(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    value = re.sub(r'[^a-z0-9]', '', value)
    aliases = {'betnijahlaneyhamilton': 'betnijahlaney', 'skylar digginssmith': 'skylardiggins'}
    return aliases.get(value, value)


def season_stats(records):
    """Aggregate regular-season stats only; MIP uses the immediately prior year."""
    boxes = records[records.season_type == 2].copy()
    if boxes.empty:
        return pd.DataFrame()
    first_season = boxes.groupby('player_id').season.min()
    keys = ['season', 'player_id']
    average = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'minutes', 'game_score']
    stats = boxes.groupby(keys)[average].mean()
    stats['games'] = boxes.groupby(keys).game_id.nunique()
    stats['player_name'] = boxes.groupby(keys).player_name.last()
    stats['team_id'] = boxes.groupby(keys).team_id.last()
    stats['player_elo'] = boxes.groupby(keys).player_elo.last()
    stats['last_game'] = boxes.groupby(keys).date.max()
    stats['qualifying_games'] = boxes.groupby(keys).minutes.apply(lambda x: int((x >= 20).sum()) + min(2, int(((x >= 15) & (x < 20)).sum())))
    sums = boxes.groupby(keys)[['points', 'field_goals_attempted', 'free_throws_attempted']].sum()
    stats['true_shooting'] = sums.points / (2*(sums.field_goals_attempted + .44*sums.free_throws_attempted)).replace(0,np.nan)
    team_games = boxes.drop_duplicates(['season', 'team_id', 'game_id']).groupby(['season','team_id']).agg(team_games=('game_id','size'), team_win_pct=('won','mean'))
    stats = stats.reset_index().merge(team_games.reset_index(), on=['season','team_id'], how='left')
    stats['games_share'] = (stats.games/stats.team_games).clip(upper=1)
    stats['rookie_inferred'] = (stats.season == stats.player_id.map(first_season)) & (stats.season > boxes.season.min())
    prev_columns = ['points','rebounds','assists','minutes','game_score']
    previous = stats[['season','player_id']+prev_columns].copy()
    previous['season'] += 1
    previous = previous.rename(columns={c: f'previous_{c}' for c in prev_columns})
    stats = stats.merge(previous, on=['season','player_id'], how='left', validate='one_to_one')
    stats['has_previous_season'] = stats.previous_points.notna()
    for stat in prev_columns:
        stats[f'{stat}_gain'] = (stats[stat]-stats[f'previous_{stat}']).where(stats.has_previous_season, 0)
    stats['name_key'] = stats.player_name.map(name_key)
    return stats.fillna({'true_shooting':0.})


def award_dataset(stats, history, award):
    """Only labelled seasons, with all historical winner matches validated."""
    history = history[history.award == award].copy()
    history['name_key'] = history.player_name.map(name_key)
    result = stats.copy()
    result['winner'] = 0
    result['previous_awards'] = [int(((history.name_key == row.name_key) & (history.season < row.season)).sum()) for row in result.itertuples()]
    valid_years = []
    for season, winners in history.groupby('season'):
        available = result[result.season == season]
        if available.empty:
            continue
        expected = set(winners.name_key)
        missing = expected - set(available.name_key)
        if missing:
            raise ValueError(f'{award} {season}: historical winners not matched to player stats: {sorted(missing)}')
        valid_years.append(season)
        result.loc[(result.season == season) & result.name_key.isin(expected), 'winner'] = 1
    return result[result.season.isin(valid_years)].copy()


def candidates(frame, award):
    frame = frame[frame.games >= 5].copy()
    if award == 'ROY':
        frame = frame[frame.rookie_inferred]
    elif award == 'MIP':
        frame = frame[frame.has_previous_season & ~frame.rookie_inferred]
    return frame


def fit_model(training):
    # Balance each season and the rare winners. No future seasons enter fitting.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.2, class_weight='balanced', max_iter=3000, random_state=42))
    weights = 1/training.groupby('season').season.transform('size')
    model.fit(training[BASE], training.winner, logisticregression__sample_weight=weights*len(training)/weights.sum())
    return model


def walk_forward(data, award):
    rows = []
    years = sorted(data.season.unique())
    for year in years[3:]:
        train = candidates(data[data.season < year], award)
        test = candidates(data[data.season == year], award)
        if train.winner.nunique() < 2 or test.empty or test.winner.sum() == 0:
            continue
        model = fit_model(train)
        test = test.assign(score=model.decision_function(test[BASE])).sort_values('score', ascending=False)
        rows.append(dict(award=award, season=int(year), top_pick=test.iloc[0].player_name,
                         top1_hit=int(test.iloc[0].winner), top3_hit=int(test.head(3).winner.any()),
                         candidates=len(test), train_through=int(train.season.max())))
    return rows


def run_awards(records, league, target_season, verbose=False):
    stats = season_stats(records)
    history = pd.read_csv(Path(__file__).with_name('award_history.csv'))
    history = history[history.league == league].copy()
    if stats.empty:
        print('Awards: no regular-season player stats available yet.')
        return pd.DataFrame()
    stats.to_csv('player_season_stats.csv', index=False)
    latest = int(stats.season.max())
    target_season = int(target_season)
    preseason = target_season > latest
    print(f'\n{league.upper()} {target_season} AWARDS - ' + ('preseason watchlist using prior-season stats' if preseason else 'season-to-date rankings'))
    print('Model scores rank candidates; they are NOT win probabilities. No injury forecast or official eligibility certification.')
    print('ROY cohort is inferred from first recorded league appearance since 2018; returning veterans may need manual eligibility review.')
    results, evaluations = [], []
    for award in AWARDS:
        labelled = award_dataset(stats, history, award)
        evaluations.extend(walk_forward(labelled[labelled.season < target_season], award))
        training = candidates(labelled[labelled.season < target_season], award)
        # Final labels from the target season are never used in its prediction.
        source = stats[stats.season == (latest if preseason else target_season)].copy()
        if preseason and award == 'ROY':
            print('ROY: awaiting rookie regular-season appearances; no invented preseason player statistics.')
            continue
        source = candidates(source, award)
        if source.empty or training.season.nunique() < 3 or training.winner.nunique() < 2:
            print(f'{award}: insufficient candidate/history data to train reliably.')
            continue
        award_history = history[(history.award == award) & (history.season < target_season)]
        source['previous_awards'] = [sum(award_history.player_name.map(name_key) == name) for name in source.name_key]
        model = fit_model(training)
        # Relative logit score anchored to the top candidate, not normalized odds.
        score = model.decision_function(source[BASE])
        source['model_score'] = 100/(1+np.exp(np.clip(score.max()-score, -700, 700))) * 2
        source = source.sort_values('model_score', ascending=False)
        source['rank'] = np.arange(1,len(source)+1)
        source['award'], source['target_season'] = award, target_season
        source['mode'] = 'preseason_prior_stats' if preseason else 'season_to_date'
        source['stats_through'] = str(records.date.max())
        source['training_through'] = int(training.season.max())
        if league == 'nba' and target_season >= 2024 and award != 'ROY':
            source['eligibility_note'] = '65-game eligibility pending' if preseason else np.where(source.qualifying_games >= 65, 'Standard 65-game threshold met', 'Below standard threshold; remaining games/exceptions not assessed')
        else:
            source['eligibility_note'] = 'Not independently verified'
        results.append(source)
        print(f'\n{award} (trained on {training.season.nunique()} prior award seasons):')
        print(source[['rank','player_name','model_score']].head(10 if verbose else 3).to_string(index=False, float_format=lambda x:f'{x:.1f}'))
    evaluation = pd.DataFrame(evaluations)
    evaluation.to_csv('award_backtest.csv', index=False)
    if not evaluation.empty:
        print('\nHistorical walk-forward checks (full-season stats, not preseason accuracy):')
        for award, group in evaluation.groupby('award', sort=False):
            print(f'{award}: top pick correct {group.top1_hit.sum()}/{len(group)} seasons; actual winner in top 3 {group.top3_hit.sum()}/{len(group)}.')
    output = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    output.to_csv('award_predictions.csv', index=False)
    print('Full rankings: award_predictions.csv; historical checks: award_backtest.csv.')
    return output


if __name__ == '__main__':
    import argparse
    from league_config import LEAGUE
    parser = argparse.ArgumentParser(description='Award rankings from refreshed player data.')
    parser.add_argument('--season', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if not Path('player_game_stats.csv').exists():
        raise SystemExit(f'Run python {LEAGUE}_predictor.py first to refresh player data.')
    records = pd.read_csv('player_game_stats.csv', parse_dates=['date'])
    metadata = __import__('json').loads(Path('data_refresh.json').read_text())
    run_awards(records, LEAGUE, args.season or metadata['target_season'], args.verbose)
