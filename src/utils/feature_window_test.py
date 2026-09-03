from pipeline.extract import read_csv
import pandas as pd

df = read_csv()
windows = [12, 20]

for w in windows:
    df[f"rolling_mean_{w}"] = (
        df["log_return"]
        .rolling(window=w)
        .mean()
    )

    df[f"rolling_volatility_{w}"] = (
        df["log_return"]
        .rolling(window=w)
        .std()
    )   

threshold = df["log_return"].abs().quantile(0.99)

events = df[
    df["log_return"].abs() >= threshold
][["Date", "log_return", "range_pct"]]

print(events.head(20))

event_time = events.iloc[0]["Date"]

window = df[
    (df["Date"] >= event_time - pd.Timedelta(minutes=60)) &
    (df["Date"] <= event_time + pd.Timedelta(minutes=60))
][[
    "Date",
    "log_return",
    "rolling_volatility_12",
    "rolling_volatility_20"
]]

print(window)

# Select the first 20 extreme movement events
top_events = (
    df.loc[df["log_return"].abs().nlargest(20).index]
    .sort_values("Date")
)

print(
    top_events[
        ["Date", "log_return", "range_pct"]
    ].to_string(index=False)
)

results = []

for _, event in top_events.iterrows():
    event_time = event["Date"]

    # Find the event row
    event_idx = df.index[df["Date"] == event_time][0]

    # 1 candle before and the event itself
    before = df.loc[event_idx - 1]
    at_event = df.loc[event_idx]

    # Maximum volatility within the next 12 candles
    after = df.loc[
        event_idx:event_idx + 12,
        ["rolling_volatility_12", "rolling_volatility_20"]
    ]

    results.append({
        "Date": event_time,
        "log_return": event["log_return"],

        "vol12_before": before["rolling_volatility_12"],
        "vol12_event": at_event["rolling_volatility_12"],
        "vol12_peak": after["rolling_volatility_12"].max(),

        "vol20_before": before["rolling_volatility_20"],
        "vol20_event": at_event["rolling_volatility_20"],
        "vol20_peak": after["rolling_volatility_20"].max(),
    })

response_df = pd.DataFrame(results)

print(response_df.to_string(index=False))

response_df["response12"] = (
    response_df["vol12_peak"] -
    response_df["vol12_before"]
)

response_df["response20"] = (
    response_df["vol20_peak"] -
    response_df["vol20_before"]
)

response_df["12_vs_20"] = (
    response_df["response12"] /
    response_df["response20"]
)

print(
    response_df[
        [
            "Date",
            "response12",
            "response20",
            "12_vs_20"
        ]
    ].to_string(index=False)
)

vol_comparison = df[
    ["rolling_volatility_12", "rolling_volatility_20"]
].dropna()

print(vol_comparison.describe())

ratio = (
    vol_comparison["rolling_volatility_12"] /
    vol_comparison["rolling_volatility_20"]
)

print(ratio.describe())

vol_comparison["abs_diff"] = (
    vol_comparison["rolling_volatility_12"]
    - vol_comparison["rolling_volatility_20"]
).abs()

print(vol_comparison["abs_diff"].describe())

ratio = (
    vol_comparison["rolling_volatility_12"]
    / vol_comparison["rolling_volatility_20"]
)

print("12 > 20 by 10%:", (ratio > 1.10).mean())
print("12 > 20 by 20%:", (ratio > 1.20).mean())
print("12 > 20 by 50%:", (ratio > 1.50).mean())

print("12 < 20 by 10%:", (ratio < 0.90).mean())
print("12 < 20 by 20%:", (ratio < 0.80).mean())
print("12 < 20 by 50%:", (ratio < 0.50).mean())

mean_comparison = df[
    ["rolling_mean_12", "rolling_mean_20"]
].dropna()

print(mean_comparison.describe())

mean_ratio = (
    mean_comparison["rolling_mean_12"].abs()
    / mean_comparison["rolling_mean_20"].abs()
)

print(mean_ratio.describe())

mean_comparison["abs_diff"] = (
    mean_comparison["rolling_mean_12"]
    - mean_comparison["rolling_mean_20"]
).abs()

print(mean_comparison["abs_diff"].describe())

df["year"] = df["Date"].dt.year
yearly = df.groupby("year").agg(
    vol12_mean=("rolling_volatility_12", "mean"),
    vol20_mean=("rolling_volatility_20", "mean"),
    mean12_std=("rolling_mean_12", "std"),
    mean20_std=("rolling_mean_20", "std")
)

print(yearly)