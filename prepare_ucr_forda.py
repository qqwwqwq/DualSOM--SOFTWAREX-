"""
Standalone script to download, preprocess, and format the FordA dataset.

This script safely downloads the FordA dataset from a highly reliable GitHub
mirror (maintained by AI researcher H. I. Fawaz), bypassing the unstable UCR servers.
It processes the raw .tsv files (where the label is the first column), and
rearranges them so the label is the last column, matching the generic
DualSOM CSV ingestion pipeline.

FordA Dataset Info:
- Domain: Automotive engine noise/vibration sensors
- Sequence Length: 500
- Classes: 2 (Normal vs. Abnormal condition)
"""

import os
import urllib.request
import pandas as pd
import numpy as np

def prepare_forda(output_dir="Datas/FordA"):
    print(">>> Starting FordA Dataset Preparation...")
    os.makedirs(output_dir, exist_ok=True)

    # Highly reliable GitHub mirror for UCR datasets
    base_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    files_to_download = {
        "TRAIN": "FordA_TRAIN.tsv",
        "TEST": "FordA_TEST.tsv"
    }

    for split, filename in files_to_download.items():
        url = base_url + filename
        raw_filepath = os.path.join(output_dir, filename)

        # 1. Download the TSV file
        if not os.path.exists(raw_filepath):
            print(f">>> Downloading {split} set from reliable mirror...")
            try:
                # Add headers to act like a standard browser
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(raw_filepath, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"    [Error] Failed to download {filename}: {e}")
                return
        else:
            print(f">>> {filename} already exists, skipping download.")

        # 2. Process and reformat the data for DualSOM
        print(f"    Formatting {split} data...")
        # Read the tab-separated values
        # In this dataset, the FIRST column contains the labels (-1 or 1)
        df = pd.read_csv(raw_filepath, sep='\t', header=None)

        labels = df.iloc[:, 0].astype(int)
        features = df.iloc[:, 1:]

        # Optional: Normalize labels from {-1, 1} to {0, 1} for cleaner metric tracking
        if -1 in labels.values:
            labels = labels.replace(-1, 0)

        # Reconstruct DataFrame with label safely at the END
        formatted_df = pd.concat([features, labels], axis=1)

        # Assign generic column names
        feature_cols = [f"sensor_t{i}" for i in range(features.shape[1])]
        formatted_df.columns = feature_cols + ["label"]

        # 3. Save as standard CSV
        csv_filename = "train_data.csv" if split == 'TRAIN' else "test_data.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        formatted_df.to_csv(csv_path, index=False)
        print(f"    Saved standard CSV to: {csv_path} (Shape: {formatted_df.shape})")

        # Cleanup the raw .tsv to keep the directory clean
        os.remove(raw_filepath)

    print("\n>>> FordA preparation complete! Your data is ready for DualSOM.")

if __name__ == "__main__":
    prepare_forda()