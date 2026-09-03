from extract import read_data
from transform import transform_data

df = read_data()
df = transform_data(df)

OUTPUT_PATH = "data/processed/xauusd_5m_processed.parquet"

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(f"Saved transformed dataset to: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")