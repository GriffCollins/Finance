import os
import requests
import zipfile
from datetime import datetime, timedelta

BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/"
ZIP_DIR = "klines_zip/"
CSV_DIR = "klines_csv/"

os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def download_zip(date):
    date_str = date.strftime("%Y-%m-%d")
    fname = f"BTCUSDT-1m-{date_str}.zip"
    url = BASE_URL + fname
    save_path = os.path.join(ZIP_DIR, fname)

    if os.path.exists(save_path):
        return save_path

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        return None

    with open(save_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    return save_path

def unzip_to_csv(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            # Example: ETHUSDT-1m-2021-01-01.csv
            out_path = os.path.join(CSV_DIR, name)
            if not os.path.exists(out_path):
                z.extract(name, CSV_DIR)
            return out_path

start = datetime(2017, 8, 17)
end = datetime.today()

d = start
while d <= end:
    zip_path = download_zip(d)
    if zip_path:
        csv_path = unzip_to_csv(zip_path)
        print("Extracted:", csv_path)
    d += timedelta(days=1)