import argparse
import nba_predictor as engine
from model_output import show_saved_diagnostics
from console_output import show_predictions

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from nba_predictor import FEATURE_COLUMNS as FEATURES


TARGET = "home_winner"
DATA_PATH = "nba_features.csv"
PREDICTIONS_PATH = "predictions.csv"


def load_data(path=DATA_PATH):
    """Load completed games and sort them chronologically."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit("Run python nba_predictor.py --predict-next first to build nba_features.csv.")

    df = df[df["completed"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["neutral_site"] = df["neutral_site"].astype(int)
    df = df.sort_values("date")

    return df


def create_models():
    """Create the models we want to compare."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=857
        ),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train a model and calculate evaluation metrics."""
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "log_loss": log_loss(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
    }


def train_time_split(df, test_fraction=0.20, verbose=False):
    """
    Train and evaluate models using a chronological train/test split.

    The final selected model is then retrained on all historical games.
    """
    split_index = int(len(df) * (1 - test_fraction))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    models = create_models()

    results = {}
    best_model = list(models.values())[0]
    best_model_name = None
    best_accuracy = -1

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "log_loss": log_loss(y_test, probabilities),
            "brier_score": brier_score_loss(y_test, probabilities),
        }

        results[model_name] = metrics

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_model = model
            best_model_name = model_name

    if verbose:
        print(f"Train games: {len(train_df)}")
        print(f"Test games:  {len(test_df)}")

        print("\nResults:")
        for model_name, metrics in results.items():
            print(
                f"{model_name}: "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"log_loss={metrics['log_loss']:.3f}, "
                f"brier_score={metrics['brier_score']:.3f}"
            )

    print(f"\nBest model: {best_model_name} (accuracy {best_accuracy:.3f})")

    # Retrain the winning model using all available historical data.
    best_model.fit(
        df[FEATURES],
        df[TARGET].astype(int),
    )

    return best_model, best_model_name, results


def predict_games(model, path=DATA_PATH, verbose=False):
    """Generate predictions for upcoming games."""
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    upcoming = df[(~df["completed"]) & (df["date"] >= pd.Timestamp.now(tz="UTC"))].copy()

    if upcoming.empty:
        print("\nNo upcoming games found.")
        return upcoming

    upcoming["neutral_site"] = upcoming["neutral_site"].astype(int)

    # Probability that the home team wins.
    upcoming["home_win_prob"] = model.predict_proba(
        upcoming[FEATURES]
    )[:, 1]

    upcoming["predicted_winner"] = upcoming.apply(
        get_predicted_winner,
        axis=1,
    )

    upcoming["confidence"] = upcoming["home_win_prob"].apply(
        lambda probability: max(probability, 1 - probability)
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

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    show_predictions(predictions, verbose=verbose, path=PREDICTIONS_PATH)

    return predictions


def get_predicted_winner(row):
    """Return the team with the higher predicted win probability."""
    if row["home_win_prob"] >= 0.5:
        return row["home_team"]

    return row["away_team"]


def main():
    """Train the model and generate predictions."""
    parser = argparse.ArgumentParser(description="Train and predict NBA games.")
    parser.add_argument("--verbose", action="store_true", help="Show the full upcoming schedule instead of each team's next game.")
    args = parser.parse_args()
    data = load_data()

    model, model_name, results = train_time_split(data, verbose=True)

    show_saved_diagnostics(pd.read_csv(DATA_PATH), model, FEATURES, engine)
    predict_games(model, verbose=args.verbose)


if __name__ == "__main__":
    main()