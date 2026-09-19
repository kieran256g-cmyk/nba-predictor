"""Shared prediction presentation for the NBA and WNBA projects."""

import pandas as pd


def upcoming_predictions(predictions, now=None):
    """Return future fixtures in chronological order without changing the input."""
    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    valid_teams = pd.Series(True, index=result.index)
    for column in ["away_team", "home_team"]:
        names = result[column].astype("string").str.strip().str.casefold()
        valid_teams &= names.notna() & ~names.isin(["", "tbd", "tba", "unknown", "to be determined", "to be announced"])
    return (
        result.loc[valid_teams & result["date"].notna() & (result["date"] >= cutoff)]
        .sort_values(["date", "away_team", "home_team"], kind="stable")
        .drop_duplicates(["date", "away_team", "home_team"])
        .reset_index(drop=True)
    )


def next_games_by_team(predictions, now=None):
    """Union of every team's first future fixture; shared fixtures appear once.

    An opponent can appear again when that later matchup is another team's
    first game. Skipping it would leave that other team without its next game.
    """
    upcoming = upcoming_predictions(predictions, now=now)
    seen = set()
    selected = []
    for index, game in upcoming.iterrows():
        teams = {game["away_team"], game["home_team"]}
        if teams - seen:
            selected.append(index)
        seen.update(teams)
    return upcoming.loc[selected].copy()


def show_predictions(predictions, verbose=False, path="predictions.csv"):
    upcoming = upcoming_predictions(predictions)
    preview = upcoming if verbose else next_games_by_team(upcoming)
    if preview.empty:
        print("\nNo upcoming games available in this schedule. Refresh the schedule to check for new games.")
        return
    display = preview[["date", "away_team", "home_team", "predicted_winner", "confidence"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    display.columns = ["Date", "Away", "Home", "Pick", "Confidence"]
    if verbose:
        print(f"\nAll upcoming predictions ({len(preview)} games):")
    else:
        teams = set(preview["away_team"]) | set(preview["home_team"])
        print(f"\nEach team's next game ({len(teams)} teams, {len(preview)} matchups):")
    print(display.to_string(index=False))
    print(f"Full prediction results saved to {path}.")
    if not verbose and len(preview) < len(upcoming):
        print("Use --verbose to show the full upcoming schedule.")
