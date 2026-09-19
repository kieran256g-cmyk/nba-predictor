"""Compact console presentation; exported predictions retain full precision."""


def show_predictions(predictions, verbose=False, path="predictions.csv"):
    preview = predictions if verbose else predictions.head(5)
    display = preview[["date", "away_team", "home_team", "predicted_winner", "confidence"]].copy()
    display["date"] = display["date"].astype(str).str[:10]
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    display.columns = ["Date", "Away", "Home", "Pick", "Confidence"]
    print(f"\nUpcoming predictions ({len(preview)} of {len(predictions)}):")
    print(display.to_string(index=False))
    print(f"All {len(predictions)} predictions saved to {path}.")
    if len(preview) < len(predictions):
        print("Use --verbose to show all predictions.")
