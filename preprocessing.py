"""
Data Ingestion Module (Preprocessed Data Scope).

This module provides a generic and robust interface for loading preprocessed
CSV datasets into the pipeline. It establishes the foundational data structures
required by the Sparse Autoencoder and Dual-SOM models.

Core Assumptions:
    - The target dataset contains purely numerical features.
    - The data is strictly structured where columns `0` through `N-1` are the 
      features, and the final trailing column `N` contains the target class labels.
"""

import os
import pandas as pd
import numpy as np

def get_dataset(data_path):
    """
    Primary data ingestion interface connecting physical files to the pipeline.

    Securely loads custom, pre-cleaned CSV datasets from a designated local path.

    Args:
        data_path (str): The localized filepath to the preprocessed CSV file.

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
    
    print(f"\n>>> Loading preprocessed dataset from {data_path}")

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
