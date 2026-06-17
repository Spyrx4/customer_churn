import pandas as pd
import os

base_dir = r"c:\UNISKA\projects\customer_churn"
csv_path = os.path.join(base_dir, "data", "train.csv")
parquet_path = os.path.join(base_dir, "data", "train.parquet")

print("Loading CSV...")
df = pd.read_csv(csv_path)

print("Saving to Parquet...")
df.to_parquet(parquet_path, engine="pyarrow", index=False)

print(f"Done! Saved to {parquet_path}")
