from pipeline.extract import read_csv
import pandas as pd
import numpy as np

df = read_csv()

# OPTIONAL: Preprocessing
print(df.shape)
print(df.dtypes)
print(df.duplicated().sum())

# Low < Open, Close, High
print((df["High"] < df["Low"]).sum())
print((df["Open"] > df["High"]).sum())
print((df["Open"] < df["Low"]).sum())
print((df["Close"] > df["High"]).sum())
print((df["Close"] < df["Low"]).sum())

# Check for date duplication and ensure ordering
print(df["Date"].duplicated().sum())
print(df["Date"].is_monotonic_increasing)

# Check proper interval
diffs = df["Date"].diff().value_counts()
print(diffs.head(10))
