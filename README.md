# NBA Game Winner Predictor

An NBA version of the WNBA predictor, using pre-game Elo ratings, recent form,
rest, head-to-head history and optional player statistics. It compares logistic
regression and gradient boosting, then predicts upcoming game winners.

## Setup and run

Requires Python 3.10 or newer and internet access to refresh schedule data.

```powershell
git clone https://github.com/kieran256g-cmyk/nba-predictor.git
cd nba-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python nba_predictor.py --predict-next
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

The first run downloads NBA schedules, creates `nba_features.csv`, evaluates
models, and writes every upcoming prediction to `predictions.csv`. The console
shows the selected model and each team's next scheduled game, with shared matchups listed once. Elo rankings, feature
importance and full prediction tables appear only with `--verbose`:

```bash
python nba_predictor.py --predict-next --verbose
```

NBA seasons use their **ending year**: `2027` means the 2026–27 season. Defaults
cover seven seasons through the current/upcoming season. To choose explicitly:

```bash
python nba_predictor.py --seasons 2024 2025 2026 2027 --test-season 2026 --predict-next
```

At least two seasons containing completed games are needed for season-based
evaluation. Missing published seasons are reported by the data loader.

To retrain using the saved feature dataset without downloading again:

```bash
python train.py
python train.py --verbose
```

Refresh with `nba_predictor.py --predict-next` as results arrive. `train.py`
uses the saved snapshot; it cannot learn outcomes that are not in that file.

## Features and evaluation

- Elo difference, including home advantage in the rating update and seasonal regression.
- Win percentage and point differential over the last 5 and 10 completed games.
- Rest difference, neutral site, and prior head-to-head results.
- A playoff Elo adjustment derived from earlier playoff results.
- Optional player-stat differences; neutral defaults when no file is supplied.

Features are recorded before each game updates team history. Scores and the
winner are labels/outcomes, not input features. Preseason, All-Star teams and
unresolved TBD opponents are excluded; regular-season, postseason and play-in
games between NBA franchises are retained.

`nba_predictor.py` trains on earlier seasons and selects the model by accuracy
on the latest season with completed games (or `--test-season`). `train.py` uses
a chronological 80/20 split. Both report accuracy, log loss and Brier score in
verbose mode, then refit the chosen model on all completed games for forecasts.
The selection-set score is validation performance, not an independent final test.

Upcoming predictions use the information currently available. Distant games
do not simulate intervening outcomes. Elo parameters are inherited starting
values from the WNBA project and have not been tuned specifically for the NBA.

## Optional player data

Place `player_stats.csv` beside the scripts with columns `date`, `team_id`,
`points`, `rebounds`, `assists`, `steals`, `blocks`, `turnovers`, `fg_pct`,
`three_pct`, `ft_pct`, and `minutes`. Use ESPN NBA team IDs. Only rows dated
strictly before the predicted game are used. Counting statistics are summed
by team/date, percentages averaged, and the latest prior team row is used.
No injuries, roster changes or player data are fetched automatically.

## Files

- `nba_predictor.py`: NBA schedule loading, feature generation, evaluation and forecasts.
- `train.py`: retraining and forecasts from the saved feature dataset.
- `console_output.py`: compact prediction formatting.
- `test_predictor.py`: offline checks for filtering, feature timing and output.
- `nba_features.csv`, `predictions.csv`: generated locally, excluded from Git.

Run checks with `python -m unittest -v`.

Schedule data comes from the [SportsDataverse NBA datasets](https://github.com/sportsdataverse/sportsdataverse-data),
loaded through [sportsdataverse for Python](https://py.sportsdataverse.org/).
Predictions are estimates, not guarantees; source coverage and data quality affect results.

### Each team's next game

Both leagues use the same columns: Date, Away, Home, Pick, Confidence.
There is no five-game limit. Games are sorted chronologically, and the first
future fixture for every team in the loaded schedule is included. A shared
fixture appears once. An opponent can appear again when that later fixture is
another team's next game; this ensures no team's actual next game is skipped.

The next available games can be in the current season or the upcoming season.
Teams without a published upcoming fixture cannot be listed. Refresh the
schedule when new fixtures are published. Past fixtures are excluded from the
console list. Full predictions remain in `predictions.csv`; `--verbose` shows
the full upcoming schedule and detailed diagnostics.
