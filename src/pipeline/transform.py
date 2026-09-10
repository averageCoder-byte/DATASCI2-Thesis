from extract import read_data
import pandas as pd
import numpy as np

df = read_data()

# FEATURE ENGINEERING

def transform_data(df):

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["range_pct"] = (df["High"]-df["Low"])/df["Close"]
    df["body_pct"] = abs(df["Close"]-df["Open"])/df["Close"]
    df["upper_wick_pct"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / df["Close"]

    df["lower_wick_pct"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    ) / df["Close"]

    # GAP-AWARE SEGMENTATION

    gap = df["Date"].diff() > pd.Timedelta(minutes=5)
    df["segment_id"] = gap.cumsum()

    # ROLLING FEATURES

    ROLLING_WINDOW = 12

    df["rolling_volatility"] = (
        df.groupby("segment_id")["log_return"]
        .rolling(window=ROLLING_WINDOW)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_return"] = (
        df.groupby("segment_id")["log_return"]
        .rolling(window=ROLLING_WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # CLEAN INCOMPLETE FEATURES

    FEATURE_COLUMNS = [
        "log_return",
        "range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "rolling_volatility",
        "rolling_mean_return",
    ]

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(drop=True)

    # TRANSFORMATION SUMMARY

    print(f"Valid feature rows: {len(df):,}")
    print(f"Continuous segments: {df['segment_id'].nunique():,}")

    return df

















