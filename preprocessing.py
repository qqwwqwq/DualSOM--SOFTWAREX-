import pandas as pd
import numpy as np
import torch
import os
import glob

# ---------------------------------------------------------
# 1. Dataset Configuration (Label Maps)
# ---------------------------------------------------------

# Maps textual action descriptions to unique integer IDs for the WUT dataset
LABEL_MAP_WUT = {
    'Collecting': 0, 'bowing': 1, 'cleaning': 2, 'drinking': 3, 'eating': 4,
    'looking': 5, 'opening': 6, 'passing': 7, 'picking': 8, 'placing': 9,
    'pushing': 10, 'reading': 11, 'sitting': 12, 'standing': 13,
    'standing_up': 14, 'talking': 15, 'turing_front': 16, 'turning': 17, 'walking': 18
}

# Maps string-based identifiers (often from PKU dataset headers) to integer IDs
LABEL_MAP_PKU = {
    "1.0": 0, "3.0": 1, "5.0": 2, "6.0": 3, "7.0": 4,
    "9.0": 5, "11.0": 6, "13.0": 7, "22.0": 8, "25.0": 9,
    "28.0": 10, "32.0": 11, "33.0": 12, "34.0": 13, "35.0": 14,
    "42.0": 15, "44.0": 16, "47.0": 17, "49.0": 18, "51.0": 19
}

# Central registry for dataset configurations
DATASET_CONFIGS = {
    'wut': LABEL_MAP_WUT,
    'pku': LABEL_MAP_PKU
}

# ---------------------------------------------------------
# 2. Internal Logic (Private Helper Functions)
# ---------------------------------------------------------

def _read_raw_folder(folder_path):
    """
    Scans a directory for CSV files, merges them, and cleans basic formatting.
    """
    print(f"   [IO] Scanning folder: {folder_path} ...")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"Error: No .csv files found in {folder_path}")

    df_list = []
    for f in all_files:
        try:
            # Read as object initially to prevent premature type conversion errors
            df = pd.read_csv(f, dtype=object)
            df_list.append(df)
        except Exception as e:
            print(f"   Warning: Skipping file {f} ({e})")

    if not df_list: 
        raise ValueError("No data could be read from the specified directory.")

    # Combine all individual CSVs into one large DataFrame
    data = pd.concat(df_list, ignore_index=True, sort=False)
    
    # Clean column names (remove whitespace) and attempt numeric conversion
    data = data.rename(columns=lambda x: str(x).strip())
    data = data.apply(pd.to_numeric, errors='ignore')

    # Remove rows where the label is 'not specified' (common in raw sensor data)
    if data.columns[-1] in data.columns:
        label_col = data.columns[-1]
        data = data[data[label_col] != "not specified"]

    # Shuffle the dataset to ensure IID (Independent and Identically Distributed) samples
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return data


def _l2_normalize(df):
    """
    Performs L2 Normalization (Unit Norm) on a per-row basis.
    Commonly used for feature vectors to ensure scale invariance.
    """
    arr = df.to_numpy().astype(np.float32)
    # Calculate the Euclidean norm (magnitude) for each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Prevent division by zero by replacing 0 norms with a tiny epsilon
    norms[norms == 0] = 1e-12
    return pd.DataFrame(arr / norms, columns=df.columns, index=df.index)


def _preprocess_and_save(raw_folder_path, save_path, label_map):
    """
    The full preprocessing pipeline: Read -> Clean -> Normalize -> Map Labels -> Save CSV.
    """
    df = _read_raw_folder(raw_folder_path)

    # Separate features (all columns but last) and labels (last column)
    features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    labels = df.iloc[:, -1]

    # Handle missing values by filling with column means
    if features.isnull().sum().sum() > 0:
        features.fillna(features.mean(), inplace=True)

    print("   [Process] Applying L2 Normalization...")
    features = _l2_normalize(features)

    print(f"   [Process] Encoding labels (Map size: {len(label_map)})...")
    # Convert string/float labels to integer IDs using the provided map
    labels_encoded = labels.map(lambda x: label_map.get(str(x), -1))

    # Filter out samples with labels that don't exist in our dictionary
    valid_mask = (labels_encoded != -1)
    if (~valid_mask).sum() > 0:
        print(f"   [Process] Removing {(~valid_mask).sum()} rows with invalid labels")
        features = features[valid_mask]
        labels_encoded = labels_encoded[valid_mask]

    # Concatenate features and labels back together for storage
    full_df = pd.concat([features, labels_encoded.rename('label')], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)
    print(f"   [Done] Processed data saved to: {save_path}")


# ---------------------------------------------------------
# 3. Public Interface
# ---------------------------------------------------------

def get_dataset(raw_path, processed_path, dataset_name='wut', force_update=False, feature_dim=57):
    """
    Main entry point for loading data. Returns PyTorch tensors.
    
    Args:
        raw_path: Path to folder containing raw CSV files.
        processed_path: Path to save/load the cleaned CSV cache.
        dataset_name: 'wut' or 'pku' (determines label mapping).
        force_update: If True, ignores existing cache and re-runs preprocessing.
        feature_dim: Expected number of feature columns.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    target_map = DATASET_CONFIGS[dataset_name]

    print(f"\n>>> Loading Dataset [{dataset_name.upper()}]")

    # Check if we need to run preprocessing or just load from cache
    if force_update or not os.path.exists(processed_path):
        print(f"   (Cache missing or update forced: regenerating...)")
        _preprocess_and_save(raw_path, processed_path, target_map)
    else:
        print(f"   (Reading existing cache: {os.path.basename(processed_path)})")

    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV: {processed_path}. Error: {e}")

    X_part = df.iloc[:, :-1]
    y_part = df.iloc[:, -1]

    # Handle cases where the cache file labels might still be strings
    if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
        print(f"   [Warning] Labels in cache are strings, applying real-time mapping...")
        y_numeric = y_part.map(lambda x: target_map.get(str(x), -1))
        if (y_numeric == -1).all():
            raise ValueError(f"Mapping failed! Check if dataset_name='{dataset_name}' matches data labels.")
        y_part = y_numeric

    try:
        # Convert to float32 (standard for deep learning models)
        X_np = X_part.values.astype(np.float32)
        y_np = y_part.values.astype(np.float32)
    except ValueError as e:
        print(f"   [Fatal Error] Could not convert data to float: {e}")
        raise e

    # Logic to fix/reshape dimensions if they don't match expectation
    if X_np.shape[1] != feature_dim:
        try:
            X_np = X_np.reshape(-1, feature_dim)
        except ValueError:
            # If reshaping is impossible, proceed with raw shape and hope for the best
            pass

    print(f"   Dataset loaded: X={X_np.shape}, y={y_np.shape}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)
