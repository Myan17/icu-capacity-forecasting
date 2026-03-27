import pandas as pd


def naive_forecast_next_24h(df: pd.DataFrame) -> list[float]:
    """
    Very first simple baseline:
    use rolling mean of recent ICU occupancy.
    """
    if df.empty:
        return [0.0] * 24

    series = df["icu_occupied"].astype(float)

    if len(series) >= 6:
        base = series.tail(6).mean()
    else:
        base = series.mean()

    return [float(base)] * 24