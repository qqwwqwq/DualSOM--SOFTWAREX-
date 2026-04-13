"""
Data Ingestion Module (Preprocessed Data Scope).

This module provides a generic and robust interface for loading preprocessed
CSV datasets into the pipeline. It establishes the foundational data structures
required by the Sparse Autoencoder and Dual-SOM models.

Core Assumptions:
    - The target dataset contains purely numerical features.
    - The data is strictly structured where columns `0` through `N-1` are the 
      features, and the final trailing column `N` contains the target class labels.

It also retains an automated network intercept for securely downloading, 
formatting, and caching the standard MNIST benchmark for experimental replication.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def get_dataset(data_path, is_train=True, dataset_name='generic'):
    """
    Primary data ingestion interface connecting physical files to the pipeline.

    Provides automated handling for downloading the MNIST benchmark dataset 
    from OpenML or securely loads custom, pre-cleaned CSV datasets from a 
    designated local path.

    Args:
        data_path (str): The localized filepath to the preprocessed CSV file.
        is_train (bool, optional): Identifies if the caller requires the training 
                                   or testing split. Defaults to True.
        dataset_name (str, optional): Identifier for the dataset. Triggering 
                                      'mnist' bypasses local file loading. 
                                      Defaults to 'generic'.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - X_features (np.ndarray): A 2D float32 array of shape (n_samples, n_features).
            - y_labels (np.ndarray): A 1D integer array of shape (n_samples,) containing class IDs.

    Raises:
        FileNotFoundError: If the specified CSV `data_path` does not exist on the disk.
        IOError: If Pandas encounters a parsing error while reading the CSV.
        ValueError: If the feature matrix cannot be cleanly cast to a float32 array 
                    (e.g., due to uncleaned string artifacts in the feature columns).
    """
    
    # =================================================================
    # Intercept Route: Built-in MNIST Benchmark via OpenML
    # =================================================================
    if dataset_name == 'mnist':
        if is_train:
            print(f"\n>>> Loading benchmark dataset [MNIST] (Downloading if first time...)")

        # Pull the official MNIST (784 features) directly from OpenML servers
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # Normalize pixel intensities from [0, 255] to [0.0, 1.0] to prevent gradient explosion
        X = X.astype(np.float32) / 255.0  
        y = y.astype(int)

        # Apply a static, deterministic train/test split to guarantee experimental reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=10000, test_size=2000, random_state=42
        )

        return (X_train, y_train) if is_train else (X_test, y_test)

    # =================================================================
    # Primary Route: Generic Preprocessed CSV Structured Data
    # =================================================================
    print(f"\n>>> Loading preprocessed dataset [{dataset_name.upper()}] from {data_path}")

    # Validate file existence before attempting to parse
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find data file at '{data_path}'. Please check the path.")

    try:
        # Load the CSV into a DataFrame. Assumes Pandas can handle standard headers transparently.
        df = pd.read_csv(data_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV at '{data_path}'. Underlying error: {e}")

    # Isolate the feature matrix (slice all columns except the last one)
    X_part = df.iloc[:, :-1]

    # Isolate the label vector (slice strictly the last column)
    y_part = df.iloc[:, -1]

    try:
        # Enforce strict float32 typing for neural network compatibility
        X_np = X_part.values.astype(np.float32)

        # Safety Fallback Mechanism:
        # Should the preprocessed data still incorrectly retain string-based class names 
        # (e.g., "Cat", "Dog") instead of numeric IDs, factorize them automatically 
        # into stable integers to prevent catastrophic downstream PyTorch/scikit-learn crashes.
        if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
            print("   [Warning] String labels detected in the target column. "
                  "Auto-converting to numerical IDs via factorization.")
            y_part, _ = pd.factorize(y_part)

        # Handle Pandas Series extraction safely regardless of pandas version
        if hasattr(y_part, 'values'):
            y_np = y_part.values.astype(np.float32) 
        else:
            y_np = np.array(y_part, dtype=np.float32)

    except ValueError as e:
        print(f"   [Fatal Error] Could not cast feature elements to numerical arrays. "
              f"Ensure the dataset is fully preprocessed. Details: {e}")
        raise e

    print(f"   Loading complete: Dataset Shape -> Features {X_np.shape}, Labels {y_np.shape}")

    # Return features as floats, and strictly cast labels to integers for classification metrics
    return X_np, y_np.astype(int)
