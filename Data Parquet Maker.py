import pandas as pd
import glob
import os

IN_DIR = "klines_csv/"
OUT_DIR = "btcusdt_yearly/"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(f"{IN_DIR}/BTCUSDT-1m-*.csv"))

years = {}
for f in files:
    # extract year from filename ETHUSDT-1m-YYYY-MM-DD.csv
    y = f.split("-")[2]
    years.setdefault(y, []).append(f)

for y, flist in years.items():
    dfs = [pd.read_csv(f, header=None) for f in flist]
    df_y = pd.concat(dfs, ignore_index=True)
    df_y.to_parquet(f"{OUT_DIR}/BTCUSDT_1m_{y}.parquet", index=False)
    print("Saved year:", y)