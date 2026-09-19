"""
NBA Game Winner Prediction Engine
===================================

Predicts the winner of NBA games using:

- Elo ratings
    - Updated after every completed game
    - Regressed toward the mean between seasons
    - Includes home-court advantage
    - Uses margin of victory

- Recent form
    - Win percentage over the last 5 games
    - Point differential over the last 5 games
    - Win percentage over the last 10 games
    - Point differential over the last 10 games

- Rest
    - Days since each team's previous game

Data source:
    sportsdataverse NBA schedule/results data, sourced from ESPN
    and mirrored through GitHub.

Usage:
    python nba_predictor.py

    python nba_predictor.py --predict-next

The second command also predicts any upcoming unplayed games.
"""

import argparse
from datetime import datetime, timezone
from console_output import show_predictions
from typing import cast

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)

from sportsdataverse.nba import load_nba_schedule


# =============================================================================
# CONFIGURATION
# =============================================================================

# NBA seasons use their ending year: 2027 means 2026-27.
_today = datetime.now(timezone.utc)
_season_end = _today.year + (_today.month >= 7)
SEASONS = list(range(_season_end - 6, _season_end + 1))

ELO_K = 20
ELO_HOME_ADVANTAGE = 65
ELO_MEAN = 1500
SEASON_REGRESSION = 0.75

ROLLING_WINDOWS = [5, 10]

DEFAULT_REST_DAYS = 5
MAX_REST_DAYS = 10

PLAYOFF_WIN_WEIGHT = 200
PLAYOFF_POINT_DIFF_WEIGHT = 2
PLAYOFF_MAX_ADJUSTMENT = 50
PLAYOFF_SHRINKAGE_GAMES = 5

RANDOM_STATE = 42

FEATURE_OUTPUT_FILE = "nba_features.csv"
PLAYER_STATS_FILE = "player_stats.csv"

# Player statistics used by the model. The player_stats.csv file should contain
# one row per player/team/date (or player/game) with statistics available before
# the game being predicted.
PLAYER_STAT_COLUMNS = [
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "minutes",
]





# =============================================================================
# PLAYOFF PERFORMANCE
# =============================================================================


def is_playoff_game(season_type):
    """Return whether a source row represents a postseason game."""
    return str(season_type).strip().lower() in {
        "3",
        "3.0",
        "postseason",
        "playoff",
        "playoffs",
    }


def get_playoff_elo_adjustment(team_id, playoff_log):
    """Estimate a team's playoff-specific Elo adjustment from prior results."""
    history = playoff_log.get(team_id, [])

    if not history:
        return 0.0

    win_percentage = np.mean([game["won"] for game in history])
    average_point_difference = np.mean(
        [game["point_diff"] for game in history]
    )
    reliability = len(history) / (
        len(history) + PLAYOFF_SHRINKAGE_GAMES
    )

    raw_adjustment = (
        (win_percentage - 0.5) * PLAYOFF_WIN_WEIGHT
        + average_point_difference * PLAYOFF_POINT_DIFF_WEIGHT
    )

    return float(
        np.clip(
            reliability * raw_adjustment,
            -PLAYOFF_MAX_ADJUSTMENT,
            PLAYOFF_MAX_ADJUSTMENT,
        )
    )

FEATURE_COLUMNS = (
    [
        "elo_diff",
        "rest_diff",
        "neutral_site",
        "h2h_winpct_diff",
        "h2h_pdiff_diff",
    ]
    + [f"winpct_diff_{window}" for window in ROLLING_WINDOWS]
    + [f"pdiff_diff_{window}" for window in ROLLING_WINDOWS]
    # Player-stat differences
    + [f"player_{stat}_diff" for stat in PLAYER_STAT_COLUMNS]
)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_games(seasons):
    """
    Load NBA schedule/results data for the requested seasons.

    Returns:
        DataFrame containing completed and upcoming games.
    """
    print(f"Loading NBA schedule data for seasons: {seasons}...")

    games = load_nba_schedule(
        seasons=seasons,
        return_as_pandas=True,
    )
    
    columns_to_keep = [
        "game_id",
        "season",
        "season_type",
        "date",
        "status_type_completed",
        "home_id",
        "home_display_name",
        "home_score",
        "home_winner",
        "away_id",
        "away_display_name",
        "away_score",
        "away_winner",
        "neutral_site",
    ]

    games = cast(pd.DataFrame, games)
    if games.empty:
        raise ValueError("No NBA schedule data is available for the requested seasons.")
    games = games[columns_to_keep].drop_duplicates("game_id").copy()
    games["date"] = pd.to_datetime(games["date"], utc=True)

    games = remove_exhibition_games(games)
    games = remove_stale_unplayed_games(games)

    games = (
        games
        .sort_values("date")
        .reset_index(drop=True)
    )

    completed_count = games["status_type_completed"].sum()

    print(
        f"Loaded {len(games)} games total "
        f"({completed_count} completed)."
    )

    return games


def remove_exhibition_games(games):
    """Keep NBA franchises in regular season, playoffs and play-in games.

    ESPN NBA franchise IDs are 1 through 30. This excludes All-Star teams,
    international/preseason opponents and unresolved TBD matchups.
    """
    valid_teams = (
        pd.to_numeric(games["home_id"], errors="coerce").between(1, 30)
        & pd.to_numeric(games["away_id"], errors="coerce").between(1, 30)
    )
    season_types = pd.to_numeric(games["season_type"], errors="coerce")
    return games[valid_teams & season_types.isin([2, 3, 5])].copy()


def remove_stale_unplayed_games(games):
    """
    Remove games marked as unplayed that are already in the past.

    These are assumed to be data artifacts rather than genuinely upcoming
    games.
    """
    now = pd.Timestamp.now(tz="UTC")

    is_unplayed = ~games["status_type_completed"]
    is_in_the_past = games["date"] < now

    stale_games = is_unplayed & is_in_the_past

    return games[~stale_games].copy()


# =============================================================================
# ELO
# =============================================================================


def initialise_team_elo(team_id, elo_ratings):
    """Give a new team the default Elo rating."""
    if team_id not in elo_ratings:
        elo_ratings[team_id] = ELO_MEAN

    return elo_ratings[team_id]


def regress_elo_between_seasons(
    team_id,
    season,
    elo_ratings,
    last_season_seen,
):
    """
    Regress a team's Elo toward the league mean when a new season begins.
    """
    initialise_team_elo(team_id, elo_ratings)

    previous_season = last_season_seen.get(team_id)

    if (
        previous_season is not None
        and previous_season != season
    ):
        elo_ratings[team_id] = (
            ELO_MEAN
            + SEASON_REGRESSION
            * (elo_ratings[team_id] - ELO_MEAN)
        )

    last_season_seen[team_id] = season

    return elo_ratings[team_id]


def calculate_expected_home_win_probability(
    home_elo,
    away_elo,
):
    """Calculate expected home-team win probability from Elo ratings."""
    rating_difference = (
        home_elo
        + ELO_HOME_ADVANTAGE
        - away_elo
    )

    return 1 / (
        1 + 10 ** (-rating_difference / 400)
    )


def calculate_elo_change(
    home_elo,
    away_elo,
    home_won,
    margin_of_victory,
):
    """
    Calculate the Elo adjustment after a completed game.

    Uses a margin-of-victory multiplier.
    """
    expected_home = calculate_expected_home_win_probability(
        home_elo,
        away_elo,
    )

    actual_home = 1.0 if home_won else 0.0

    elo_difference = abs(home_elo - away_elo)

    mov_multiplier = (
        np.log(abs(margin_of_victory) + 1)
        * (
            2.2
            / (
                elo_difference * 0.001
                + 2.2
            )
        )
    )

    return (
        ELO_K
        * mov_multiplier
        * (actual_home - expected_home)
    )


# =============================================================================
# RECENT FORM
# =============================================================================


def get_form_features(team_id, game_log):
    """
    Calculate recent-form statistics for a team.

    Returns win percentage and average point differential for each
    configured rolling window.
    """
    history = game_log.get(team_id, [])

    features = {}

    for window in ROLLING_WINDOWS:
        recent_games = history[-window:]

        if recent_games:
            win_percentage = np.mean(
                [game["won"] for game in recent_games]
            )

            average_point_difference = np.mean(
                [
                    game["point_diff"]
                    for game in recent_games
                ]
            )
        else:
            # Neutral defaults for teams with no historical games.
            win_percentage = 0.5
            average_point_difference = 0.0

        features[f"winpct_last{window}"] = win_percentage
        features[f"pdiff_last{window}"] = (
            average_point_difference
        )

    return features


# =============================================================================
# REST
# =============================================================================


def get_rest_days(
    team_id,
    current_date,
    last_game_date,
):
    """Calculate the number of rest days since a team's previous game."""
    previous_date = last_game_date.get(team_id)

    if previous_date is None:
        return DEFAULT_REST_DAYS

    rest_days = (
        current_date - previous_date
    ).days

    return min(rest_days, MAX_REST_DAYS)

#=============================================================================
#Head to head performance
#=============================================================================

def get_head_to_head_performance(team_id, opponent_id, game_log):
    """
    Calculate head-to-head performance statistics for a team against a specific opponent.

    Returns win percentage and average point differential for the head-to-head matchups.
    """
    history = game_log.get(team_id, [])
    head_to_head_games = [
        game for game in history if game.get("opponent_id") == opponent_id
    ]

    if head_to_head_games:
        win_percentage = np.mean(
            [game["won"] for game in head_to_head_games]
        )

        average_point_difference = np.mean(
            [
                game["point_diff"]
                for game in head_to_head_games
            ]
        )
    else:
        # Neutral defaults for teams with no historical games against the opponent.
        win_percentage = 0.5
        average_point_difference = 0.0

    return {
        "head_to_head_winpct": win_percentage,
        "head_to_head_pdiff": average_point_difference,
    }

# =============================================================================
# PLAYER STATISTICS
# =============================================================================

def load_player_stats(path=PLAYER_STATS_FILE):
    """
    Load player statistics if a player_stats.csv file is available.

    Expected columns:
        date, team_id, plus the columns in PLAYER_STAT_COLUMNS.

    Optional:
        game_id, player_id, player_name.

    Statistics are aggregated by team/date. Only rows dated before a game are
    used, preventing the model from seeing statistics from the game it predicts.
    """
    try:
        stats = pd.read_csv(path)
    except FileNotFoundError:
        print(
            f"Player stats file '{path}' was not found. "
            "Player features will use neutral defaults."
        )
        return pd.DataFrame()

    required = {"date", "team_id"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(
            f"Player stats file is missing required columns: {sorted(missing)}"
        )

    stats = stats.copy()
    stats["date"] = pd.to_datetime(stats["date"])

    for column in PLAYER_STAT_COLUMNS:
        if column not in stats.columns:
            stats[column] = 0.0
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)

    # Aggregate player production to team level for each date.
    # Percentages are averaged across player rows; counting stats are summed.
    aggregation = {}
    for column in PLAYER_STAT_COLUMNS:
        aggregation[column] = "mean" if column.endswith("_pct") else "sum"

    return (
        stats.groupby(["team_id", "date"], as_index=False)
        .agg(aggregation)
        .sort_values(["team_id", "date"])
        .reset_index(drop=True)
    )


def get_player_features(team_id, current_date, player_stats):
    """
    Return the latest available team-level player statistics before current_date.
    """
    if player_stats.empty:
        return {f"player_{stat}": 0.0 for stat in PLAYER_STAT_COLUMNS}

    history = player_stats[
        (player_stats["team_id"] == team_id)
        & (player_stats["date"] < current_date)
    ]

    if history.empty:
        return {f"player_{stat}": 0.0 for stat in PLAYER_STAT_COLUMNS}

    latest_date = history["date"].max()
    latest = history[history["date"] == latest_date].iloc[-1]

    return {
        f"player_{stat}": float(latest[stat])
        for stat in PLAYER_STAT_COLUMNS
    }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def create_feature_row(
    game,
    home_elo,
    away_elo,
    home_form,
    away_form,
    home_rest,
    away_rest,
    home_head_to_head,
    away_head_to_head,
    playoff_elo_adjustment_diff,
    home_player_stats,
    away_player_stats,
):
    """Create the pre-game feature row for one game."""

    row = {
        "game_id": game["game_id"],
        "season": game["season"],
        "date": game["date"],
        "completed": game["status_type_completed"],
        "home_id": game["home_id"],
        "away_id": game["away_id"],
        "home_team": game["home_display_name"],
        "away_team": game["away_display_name"],
        "neutral_site": bool(game["neutral_site"]),

        # Elo
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": (
            home_elo
            - away_elo
            + playoff_elo_adjustment_diff
        ),
        "playoff_elo_adjustment_diff": playoff_elo_adjustment_diff,

        # Rest
        "home_rest": home_rest,
        "away_rest": away_rest,
        "rest_diff": home_rest - away_rest,

        # Head-to-head performance
        "h2h_winpct_diff": (
            home_head_to_head["head_to_head_winpct"]
            - away_head_to_head["head_to_head_winpct"]
        ),
        "h2h_pdiff_diff": (
            home_head_to_head["head_to_head_pdiff"]
            - away_head_to_head["head_to_head_pdiff"]
        ),

        # Result information
        "home_score": game["home_score"],
        "away_score": game["away_score"],
        "home_winner": game["home_winner"],
    }

    # Add individual team form features.
    for name, value in home_form.items():
        row[f"home_{name}"] = value

    for name, value in away_form.items():
        row[f"away_{name}"] = value

    # Add home-vs-away differences.
    for window in ROLLING_WINDOWS:
        row[f"winpct_diff_{window}"] = (
            home_form[f"winpct_last{window}"]
            - away_form[f"winpct_last{window}"]
        )

        row[f"pdiff_diff_{window}"] = (
            home_form[f"pdiff_last{window}"]
            - away_form[f"pdiff_last{window}"]
        )

    # Add player stat differences.
    for stat in PLAYER_STAT_COLUMNS:
        row[f"player_{stat}_diff"] = (
            home_player_stats[f"player_{stat}"]
            - away_player_stats[f"player_{stat}"]
        )

    return row


def update_game_state(
    game,
    home_elo,
    away_elo,
    elo_ratings,
    game_log,
    playoff_log,
    last_game_date,
):
    """
    Update Elo, recent-form history, and last-game dates.

    This function is only called for completed games.
    """
    home_id = game["home_id"]
    away_id = game["away_id"]

    home_won = bool(game["home_winner"])

    margin = (
        game["home_score"]
        - game["away_score"]
    )

    # -------------------------------------------------------------------------
    # Elo
    # -------------------------------------------------------------------------

    elo_change = calculate_elo_change(
        home_elo=home_elo,
        away_elo=away_elo,
        home_won=home_won,
        margin_of_victory=margin,
    )

    elo_ratings[home_id] = home_elo + elo_change
    elo_ratings[away_id] = away_elo - elo_change

    # -------------------------------------------------------------------------
    # Recent form
    # -------------------------------------------------------------------------

    game_log.setdefault(home_id, []).append(
        {
            "won": home_won,
            "point_diff": margin,
            "opponent_id": away_id,
        }
    )

    game_log.setdefault(away_id, []).append(
        {
            "won": not home_won,
            "point_diff": -margin,
            "opponent_id": home_id,
        }
    )

    if is_playoff_game(game["season_type"]):
        playoff_log.setdefault(home_id, []).append(
            {
                "won": home_won,
                "point_diff": margin,
            }
        )
        playoff_log.setdefault(away_id, []).append(
            {
                "won": not home_won,
                "point_diff": -margin,
            }
        )

    # -------------------------------------------------------------------------
    # Rest
    # -------------------------------------------------------------------------

    last_game_date[home_id] = game["date"]
    last_game_date[away_id] = game["date"]


def compute_features(games, player_stats=None):
    """
    Walk through games chronologically and create pre-game features.

    Player statistics are optional. When supplied, only the most recent
    statistics dated before each game are used.

    Important:
        Features are calculated BEFORE each game's result is incorporated,
        preventing the model from seeing the outcome it is trying to predict.

    Returns:
        feature_df: DataFrame containing one row per game.
        elo_ratings: Final Elo rating for each team.
    """
    elo_ratings = {}
    last_season_seen = {}

    if player_stats is None:
        player_stats = pd.DataFrame()

    game_log = {}
    playoff_log = {}
    last_game_date = {}

    feature_rows = []

    for _, game in games.iterrows():
        season = game["season"]

        home_id = game["home_id"]
        away_id = game["away_id"]

        # ---------------------------------------------------------------------
        # Elo
        # ---------------------------------------------------------------------

        home_elo = regress_elo_between_seasons(
            home_id,
            season,
            elo_ratings,
            last_season_seen,
        )

        away_elo = regress_elo_between_seasons(
            away_id,
            season,
            elo_ratings,
            last_season_seen,
        )

        # ---------------------------------------------------------------------
        # Recent form
        # ---------------------------------------------------------------------

        home_form = get_form_features(
            home_id,
            game_log,
        )

        away_form = get_form_features(
            away_id,
            game_log,
        )

        # Head-to-head performance, based only on prior completed meetings.
        home_head_to_head = get_head_to_head_performance(
            home_id,
            away_id,
            game_log,
        )

        away_head_to_head = get_head_to_head_performance(
            away_id,
            home_id,
            game_log,
        )

        home_playoff_adjustment = get_playoff_elo_adjustment(
            home_id,
            playoff_log,
        )
        away_playoff_adjustment = get_playoff_elo_adjustment(
            away_id,
            playoff_log,
        )
        playoff_elo_adjustment_diff = 0.0
        if is_playoff_game(game["season_type"]):
            playoff_elo_adjustment_diff = (
                home_playoff_adjustment
                - away_playoff_adjustment
            )

        # ---------------------------------------------------------------------
        # Rest
        # ---------------------------------------------------------------------

        home_rest = get_rest_days(
            home_id,
            game["date"],
            last_game_date,
        )

        away_rest = get_rest_days(
            away_id,
            game["date"],
            last_game_date,
        )

        # ---------------------------------------------------------------------
        # Player statistics
        # ---------------------------------------------------------------------

        home_player_stats = get_player_features(
            home_id,
            game["date"],
            player_stats,
        )

        away_player_stats = get_player_features(
            away_id,
            game["date"],
            player_stats,
        )

        # ---------------------------------------------------------------------
        # Create pre-game feature row
        # ---------------------------------------------------------------------

        feature_row = create_feature_row(
            game=game,
            home_elo=home_elo,
            away_elo=away_elo,
            home_form=home_form,
            away_form=away_form,
            home_rest=home_rest,
            away_rest=away_rest,
            home_head_to_head=home_head_to_head,
            away_head_to_head=away_head_to_head,
            playoff_elo_adjustment_diff=playoff_elo_adjustment_diff,
            home_player_stats=home_player_stats,
            away_player_stats=away_player_stats,
        )

        feature_rows.append(feature_row)

        # ---------------------------------------------------------------------
        # Update historical state AFTER the game
        # ---------------------------------------------------------------------

        if game["status_type_completed"]:
            update_game_state(
                game=game,
                home_elo=home_elo,
                away_elo=away_elo,
                elo_ratings=elo_ratings,
                game_log=game_log,
                playoff_log=playoff_log,
                last_game_date=last_game_date,
            )

    feature_df = pd.DataFrame(feature_rows)

    return feature_df, elo_ratings


# =============================================================================
# MODEL TRAINING
# =============================================================================


def create_models():
    """Create the models used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train a model and return its evaluation metrics."""
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "log_loss": log_loss(
            y_test,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities,
        ),
    }


def get_feature_importance(model):
    """Extract feature importance from the trained model."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_,
            index=FEATURE_COLUMNS,
        ).sort_values(
            ascending=False
        )

    if hasattr(model, "coef_"):
        return pd.Series(
            model.coef_[0],
            index=FEATURE_COLUMNS,
        ).sort_values(
            key=abs,
            ascending=False,
        )

    return None


def train_and_evaluate(feature_df, verbose=False, test_season=None):
    """
    Evaluate on the latest available season, or an explicit test season.

    After evaluation, refit the selected model on all completed games for forecasting.
    """
    completed_games = feature_df[
        feature_df["completed"]
    ].copy()

    completed_games["neutral_site"] = (
        completed_games["neutral_site"].astype(int)
    )

    if completed_games.empty:
        raise ValueError("No completed NBA games available for training.")
    if test_season is None:
        test_season = int(completed_games["season"].max())

    train_games = completed_games[
        completed_games["season"] < test_season
    ]
    test_games = completed_games[
        completed_games["season"] == test_season
    ]

    if train_games.empty:
        raise ValueError("No training games found before the test season.")

    if test_games.empty:
        raise ValueError(f"No completed games found for {test_season}.")

    X_train = train_games[FEATURE_COLUMNS]
    y_train = train_games["home_winner"].astype(int)
    X_test = test_games[FEATURE_COLUMNS]
    y_test = test_games["home_winner"].astype(int)

    models = create_models()

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

    best_model = None
    best_model_name = None
    best_accuracy = -1

    for model_name, model in models.items():
        metrics = evaluate_model(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        if verbose:
            print(f"\n{model_name}:")
            print(
                f"  Accuracy:    {metrics['accuracy']:.3f}"
            )
            print(
                f"  Log Loss:    {metrics['log_loss']:.3f}"
            )
            print(
                f"  Brier Score: {metrics['brier_score']:.3f}"
            )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_model = model
            best_model_name = model_name

    # -------------------------------------------------------------------------
    # Home-team baseline
    # -------------------------------------------------------------------------

    home_baseline = accuracy_score(
        y_test,
        np.ones(len(y_test)),
    )

    if verbose:
        print(
            f"\nBaseline "
            f"(always pick home team): "
            f"{home_baseline:.3f}"
        )

    print(
        f"\nBest model: {best_model_name} "
        f"({test_season} validation accuracy {best_accuracy:.3f})"
    )

    # -------------------------------------------------------------------------
    # Evaluation is finished; refit on all known outcomes for future forecasts.
    # -------------------------------------------------------------------------

    if best_model is None:
        raise RuntimeError("No model was available for training.")

    best_model.fit(completed_games[FEATURE_COLUMNS], completed_games["home_winner"].astype(int))

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    feature_importance = get_feature_importance(
        best_model
    )

    if verbose and feature_importance is not None:
        print("\nFeature importance:")
        print(
            feature_importance.to_string()
        )

    return best_model, best_model_name


# =============================================================================
# PREDICTIONS
# =============================================================================


def predict_upcoming(feature_df, model, verbose=False):
    """Predict the outcome of all upcoming games."""
    upcoming = feature_df[
        ~feature_df["completed"]
    ].copy()

    if upcoming.empty:
        print(
            "\nNo upcoming (unplayed) games "
            "found in the loaded seasons."
        )
        return None

    upcoming["neutral_site"] = (
        upcoming["neutral_site"].astype(int)
    )

    probabilities = model.predict_proba(
        upcoming[FEATURE_COLUMNS]
    )[:, 1]

    upcoming["home_win_prob"] = probabilities

    upcoming["predicted_winner"] = np.where(
        probabilities >= 0.5,
        upcoming["home_team"],
        upcoming["away_team"],
    )

    upcoming["confidence"] = np.maximum(
        probabilities,
        1 - probabilities,
    )

    output_columns = [
        "date",
        "away_team",
        "home_team",
        "predicted_winner",
        "home_win_prob",
        "confidence",
    ]

    predictions = (
        upcoming[output_columns]
        .sort_values("date")
    )

    predictions.to_csv("predictions.csv", index=False)
    show_predictions(predictions, verbose=verbose)

    return predictions


# =============================================================================
# POWER RANKINGS
# =============================================================================


def build_team_name_lookup(feature_df):
    """Build a mapping from team ID to team name."""
    team_names = {}

    for _, row in feature_df.iterrows():
        team_names[row["home_id"]] = row["home_team"]
        team_names[row["away_id"]] = row["away_team"]

    return team_names


def show_power_rankings(
    elo_ratings,
    feature_df,
):
    """Display current Elo-based power rankings."""
    team_names = build_team_name_lookup(
        feature_df
    )

    rankings = (
        pd.Series(elo_ratings)
        .rename("elo")
        .sort_values(ascending=False)
    )

    rankings = rankings.rename(
        index=team_names
    )

    print("\n" + "=" * 60)
    print("CURRENT ELO POWER RANKINGS")
    print("=" * 60)

    print(
        rankings.round(1).to_string()
    )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NBA game winner prediction engine."
    )

    parser.add_argument(
        "--predict-next",
        action="store_true",
        help="Predict upcoming unplayed games (also enabled by default).",
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Show the full upcoming schedule instead of each team's next game.",
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS,
                        help="Season ending years, e.g. 2025 2026 2027.")
    parser.add_argument("--test-season", type=int, default=None,
                        help="Validation season; defaults to latest season with completed games.")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run the complete prediction pipeline."""
    args = parse_arguments()

    # 1. Load historical and upcoming games.
    games = load_games(args.seasons)

    # 2. Load player statistics and generate pre-game features.
    player_stats = load_player_stats()
    feature_df, elo_ratings = compute_features(
        games,
        player_stats=player_stats,
    )

    # 3. Train and evaluate models.
    model, model_name = train_and_evaluate(
        feature_df, verbose=True, test_season=args.test_season
    )

    # 4. Display current Elo rankings.
    show_power_rankings(elo_ratings, feature_df)

    # 5. Optionally predict upcoming games.
    predict_upcoming(feature_df, model, verbose=args.verbose)

    # 6. Save feature data.
    feature_df.to_csv(
        FEATURE_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved feature dataset to "
        f"{FEATURE_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()