"""
Data Ingestion Module (Preprocessed Data Scope).

This module provides a generic and robust interface for loading preprocessed
CSV datasets into the pipeline. It assumes that the target dataset contains
numerical features in all columns except the trailing column, which must contain
the target class labels.

It also retains an automated intercept for safely downloading and formatting
the standard MNIST benchmark for replication purposes.
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def get_dataset(data_path, is_train=True, dataset_name='generic'):
    """
    Primary data ingestion interface connecting physical files to the pipeline.

    Provides automated handling for downloading the MNIST benchmark dataset
    or securely loads custom cleaned CSV datasets from the designated path.

    Args:
        data_path (str): The localized path to the preprocessed CSV file.
        is_train (bool): Identifies if the caller requires the training or testing split.
        dataset_name (str): Identifier for the dataset (triggers specific loading logic).

    Returns:
        tuple: (X_features as a float32 numpy array, y_labels as an integer numpy array).
    """
    # === Intercept Route: Built-in MNIST Benchmark ===
    if dataset_name == 'mnist':
        if is_train:
            print(f"\n>>> Loading dataset [MNIST] (Downloading if first time...)")

        # Pull MNIST directly from OpenML
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0  # Apply min-max normalization to image pixels
        y = y.astype(int)

        # Apply a static train/test split to guarantee experimental reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=2000, random_state=42)

        return (X_train, y_train) if is_train else (X_test, y_test)

    # === Primary Route: Generic Preprocessed CSV Structured Data ===
    print(f"\n>>> Loading preprocessed dataset [{dataset_name.upper()}] from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find data file at {data_path}.")

    try:
        # Load the CSV. Assumes pandas can handle headers transparently.
        df = pd.read_csv(data_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV: {data_path}. Error: {e}")

    # Isolate feature matrix (all columns except the last sequential one)
    X_part = df.iloc[:, :-1]

    # Isolate label vector (strictly the last column)
    y_part = df.iloc[:, -1]

    try:
        X_np = X_part.values.astype(np.float32)

        # Safety Fallback: Should the preprocessed data still incorrectly retain string labels,
        # factorize them automatically into stable integers to prevent catastrophic downstream crashes.
        if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
            print("   [Warning] String labels detected in preprocessed data. Auto-converting to numerical IDs.")
            y_part, _ = pd.factorize(y_part)

        y_np = y_part.values.astype(np.float32) if hasattr(y_part, 'values') else np.array(y_part, dtype=np.float32)

    except ValueError as e:
        print(f"   [Fatal Error] Could not parse feature elements as float arrays: {e}")
        raise e

    print(f"   Loading complete: Dataset Shape -> Features {X_np.shape}, Labels {y_np.shape}")

    return X_np, y_np.astype(int)