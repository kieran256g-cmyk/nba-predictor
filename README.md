# NBA Game and Player Predictor

Predicts game winners using team Elo, form, rest, head-to-head history,
automatically refreshed player statistics and custom individual Elo ratings.
Also ranks candidates for **MVP, Rookie of the Year, Defensive Player of the Year
and Most Improved Player**, using historical award winners.

## Install and run

Python 3.10+ and internet access are required for a fresh run.

```powershell
git clone https://github.com/kieran256g-cmyk/nba-predictor.git
cd nba-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

On macOS/Linux activate with `source .venv/bin/activate` instead.
**Press Run on either `train.py` or `nba_predictor.py`: both now refresh team
schedules, player box scores and roster snapshots together**, rebuild features,
update player ratings, train game models and refresh the awards rankings.
The initial download covers seasons from 2018 onward and can take a few minutes.

```bash
python nba_predictor.py
python nba_predictor.py --verbose
python nba_predictor.py --offline
python train.py --cached-features
python awards_predictor.py
```

`--verbose` shows every upcoming game and more award candidates. Normal output
includes model comparisons, the probability confidence check, feature importance,
team Elo rankings, ten player ratings and each team's next game (shared matchups
once), followed by three candidates per award. `--predict-next` still works but
is no longer necessary.

`--offline` explicitly reuses downloaded releases for the full pipeline.
`--cached-features` is the old saved-feature game-model-only workflow; it does not
refresh player data or awards. Old feature CSVs must be rebuilt once after upgrading.
Network failures stop a fresh run rather than silently claiming cached data is new.

Use `--seasons` to choose season years and `--test-season` to choose the game
validation season. NBA years identify the ending year (2027 = 2026–27); WNBA
years identify the calendar year. Keep 2018 as a warm-up year for the bundled
award history and rookie inference. Shorter history reduces what can be trained.

## Player statistics

SportsDataverse player box scores are matched to completed scheduled games.
Duplicate appearances, DNPs, games without final scores and unresolved opponents
do not become player updates. Points, rebounds, assists, steals, blocks, turnovers,
minutes and shooting percentages enter the game model as home-minus-away features.

For each player, rolling averages use the last **5 and 10 played games**.
The historical team projection uses players seen in its last five games, excluding
players subsequently observed with another team. Player counting-stat projections
are scaled to one team's regulation minutes (240 NBA / 200 WNBA); shooting
percentages use total makes divided by attempts, not an average of percentages.
The existing `player_*_diff` features retain the five-game projection; windowed
`player_*_diff_5` and `player_*_diff_10` columns are also exported.

The next game's player features are taken **before** that game updates history.
Current roster snapshots are applied only to unplayed forecasts, never to historical
training rows. Players keep their histories and ratings across trades. New players
start neutral, and `player_history_coverage_diff` identifies incomplete history.
Expected minutes come from prior appearances; injuries, confirmed lineups and
future workload changes are not forecast. Roster releases can lag transactions.

## Individual player Elo adjustment

Each player begins at **1500**. After a completed appearance, the rating receives:

- An outcome update: `20 × participation × (win − expected_win)`.
- A box-score adjustment: `0.05 × participation × (performance_target − old_rating)`.

Participation is minutes divided by regulation game length, capped at one.
Expected win uses each side's pre-game, minutes-weighted player ratings and
a 65-point home advantage. The box-score target is
`1500 + clip((GameScore_per_36 − 12) × 12, −200, 200)`; Game Score uses points,
shooting, rebounds, assists, steals, blocks, turnovers and fouls. Ratings regress
25% toward 1500 when a player enters a new season.

`player_elo_adjustment = player_elo − 1500`. The game model learns a separate
`player_elo_adjustment_diff` from projected minutes-weighted team averages.
It is not added a second time to the original team Elo calculation.
These are **custom, untuned Elo-style ratings**, not official ratings or causal
estimates of individual impact. Last-game minutes are used only in the post-game
rating update; forecast weights use prior appearances.

## Awards model

`award_history.csv` contains sourced winner labels: NBA 2019–2026 and WNBA
2019–2025, including both 2025 WNBA DPOY winners. Each award has a regularized
logistic ranking model fitted to earlier seasons only. Features include regular-season
per-game production, efficiency, team win rate, availability, player Elo, previous
awards and changes from the immediately preceding season. Postseason stats are
excluded from award season aggregates.

ROY candidates are inferred from their first recorded league season since 2018;
the first dataset season is excluded. Returning veterans absent from the earlier
data can be misclassified, so this is not official rookie eligibility certification.
MIP candidates must have a prior-season comparison and cannot be inferred rookies.
NBA exports also show the standard 65-game/20-minute eligibility count, including
up to two 15–19-minute appearances. Exceptions and injury grievances are not
automatically decided; eligibility notes are informational, not a final ruling.

**Model scores are relative ranking scores, not award-winning probabilities.**
The top candidate is anchored at 100; scores are not calibrated and should not be
compared between awards. Historical walk-forward checks train on earlier seasons
and report top-pick and top-three hits for later seasons. Those checks use full-season
stats, so they are not evidence of preseason forecast accuracy. Sample sizes are
small and the results vary substantially between awards.

If the target season has not started, MVP/DPOY/MIP output is explicitly a preseason
watchlist based on the last season's statistics. ROY waits for rookie appearances;
the model does not invent professional stats for unplayed rookies. During a season,
rankings update with every data refresh. To inspect another target season from the
saved data, use `python awards_predictor.py --season YEAR`.

Winner labels are a reviewed snapshot, not scraped blindly on each run. Add newly
announced winners to `award_history.csv` with a source URL; unmatched winner names
raise an error instead of silently becoming negative training labels.

## Game evaluation and confidence

The refreshed pipeline evaluates on the latest season with completed games (or
`--test-season`) using only earlier seasons for fitting. It compares logistic
regression with feature scaling and gradient boosting. After choosing a model,
it refits on all completed games for forecasts. `--cached-features` instead retains
the chronological 80/20 evaluation path.

The confidence table checks home and away picks on held-out validation games
before refitting: confidence band, sample size, wins, average predicted chance,
actual win rate and percentage-point gap. These same games select the model,
so this is a validation diagnostic rather than an independent final test.
The new player features are not assumed to improve accuracy; inspect these results.

## Generated files

| File | Contents |
|---|---|
| `nba_features.csv` | Pre-game team and player features |
| `predictions.csv` | Full upcoming game predictions |
| `player_game_stats.csv` | Played appearances and post-game player Elo changes |
| `player_ratings.csv` | All observed players, last-game ratings, current roster flags |
| `player_season_stats.csv` | Regular-season player totals/averages for awards |
| `award_predictions.csv` | Candidate ranks, scores, eligibility notes and training dates |
| `award_backtest.csv` | Historical season-by-season award checks |
| `data_refresh.json` | Refresh mode, timestamp, target season and data coverage |
| `data/cache/` | Downloaded team, player and roster releases for offline runs |

Generated files are local and excluded from Git. Run tests with `python -m unittest -v`.
They cover future-data isolation, rating updates, trades, shooting aggregation,
award labels, co-winners, season splits and the existing output/confidence checks.

## Sources

- [SportsDataverse datasets](https://github.com/sportsdataverse/sportsdataverse-data)
  and [Python package](https://py.sportsdataverse.org/), sourced from ESPN.
- Official NBA and WNBA award-history pages and announcements, linked on every
  row of `award_history.csv`.
- [NBA eligibility summary](https://cms.nba.com/wp-content/uploads/sites/4/2024/11/2024-25-CBA-101.pdf).

Prediction quality depends on source coverage, lineup changes and model assumptions.
