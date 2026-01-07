#!/usr/bin/env python3
"""
Fast Mapbox satellite image fetcher (optimized for speed)
"""

import os
import io 
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
TRAIN_IMG_DIR = "data/images/train"
TEST_IMG_DIR = "data/images/test"

IMG_ZOOM = 17
IMG_SIZE = (256, 256)  # bumped up to 256 for better model input
MAX_WORKERS = 16  # parallel downloads

MAPBOX_ACCESS_TOKEN = "pk.eyJ1IjoiYXl1dXNoaHNoIiwiYSI6ImNtanoyd3Y4bTYzYXMzZnM1eTc4YWVwbHEifQ.CW6_GKnBSfcjvhkZVY-AvQ"

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)


def build_mapbox_static_url(lat, lon, token):
    width, height = IMG_SIZE
    base = "https://api.mapbox.com/styles/v1/mapbox"
    return (
        f"{base}/satellite-v9/static/"
        f"{lon},{lat},{IMG_ZOOM},0,0/"
        f"{width}x{height}"
        f"?access_token={token}"
    )


def download_image(args):
    """Single image downloader - returns (prop_id, success)"""
    prop_id, lat, lon, out_dir, token = args
    out_path = os.path.join(out_dir, f"{prop_id}.png")
    
    if os.path.exists(out_path):
        return prop_id, True
    
    if pd.isna(lat) or pd.isna(lon):
        return prop_id, False

    url = build_mapbox_static_url(lat, lon, token)

    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            # Verify it's a valid image
            img = Image.open(io.BytesIO(resp.content))
            img.verify()  # validate image
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return prop_id, True
        else:
            print(f"Failed {prop_id}: HTTP {resp.status_code}")
            return prop_id, False
    except Exception as e:
        print(f"Error {prop_id}: {e}")
        return prop_id, False


def download_for_csv(csv_path, out_dir, token, max_workers=MAX_WORKERS):
    df = pd.read_csv(csv_path)
    
    # Prepare args for parallel execution
    args_list = []
    for _, row in df.iterrows():
        args_list.append((row["id"], row["lat"], row["long"], out_dir, token))
    
    success, fail = 0, 0
    
    # Parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(args_list), desc=f"Fetching {os.path.basename(csv_path)}"):
            prop_id, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1
    
    print(f"Done {csv_path}: Success={success}, Fail={fail}")


if __name__ == "__main__":
    print("=== FAST SATELLITE IMAGE FETCHER ===")
    
    # Download train
    print("\n1. Train images...")
    download_for_csv(TRAIN_CSV_PATH, TRAIN_IMG_DIR, MAPBOX_ACCESS_TOKEN)
    
    # Download test  
    print("\n2. Test images...")
    download_for_csv(TEST_CSV_PATH, TEST_IMG_DIR, MAPBOX_ACCESS_TOKEN)
    
    print("✅ COMPLETE")
