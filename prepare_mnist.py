"""
Standalone script to download, preprocess, and format the MNIST dataset.

This script downloads the standard MNIST benchmark, scales the pixel values,
splits the data into training and testing sets, and saves them as CSV files
compatible with the generic DualSOM preprocessing module.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def prepare_mnist(output_dir="Datas/MNIST"):
    """Downloads MNIST, preprocesses it, and saves to CSV."""
    print(">>> Downloading MNIST dataset from OpenML (this may take a minute)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    print(">>> Normalizing pixel values to [0, 1]...")
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    print(">>> Splitting into training (10,000) and testing (2,000) sets...")
    # Static train/test split to guarantee experimental reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=2000, random_state=42)

    # Create output directory mapping to the DualSOM standard structure
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Saving formatted CSVs to {output_dir}/...")

    # Combine features and labels into DataFrames (label MUST be the last column)

    # 1. Train Data
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    train_csv_path = os.path.join(output_dir, "train_data.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"    Saved {train_csv_path} (Shape: {train_df.shape})")

    # 2. Test Data
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    test_csv_path = os.path.join(output_dir, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"    Saved {test_csv_path} (Shape: {test_df.shape})")

    print("\n>>> MNIST preparation complete! You can now use these CSVs in DualSOM.")


if __name__ == "__main__":
    prepare_mnist()
