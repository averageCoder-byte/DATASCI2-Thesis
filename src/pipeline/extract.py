import pandas as pd
from pandas import DataFrame

path = "data/raw/xauusd_m5_raw.csv"

# Extracting raw dataset
def read_data() -> DataFrame:
    data = pd.read_csv(path)
    data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop("Volume", axis=1)

    return df
