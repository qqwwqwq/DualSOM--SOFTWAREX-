"""
PyTorch-based Sparse Autoencoder for Latent Representation Learning.

This module projects high-dimensional input features into a compact,
low-dimensional latent space (Workflow Stage 2). It respects explicit JSON
directives for either training from scratch or loading pre-trained weights,
and utilizes robust Mini-Batch training for optimal gradient convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

class SparseAutoencoder(nn.Module):
    """
    Feed-forward Neural Network Architecture for the Sparse Autoencoder.
    Employs Batch Normalization, CELU activations, and a symmetric encoder-decoder structure.
    """
    def __init__(self, input_dim=57):
        """
        Args:
            input_dim (int): Dimensionality of the raw input feature space.
        """
        super(SparseAutoencoder, self).__init__()

        # Encoder mapping: Input -> 72 -> 36 (Latent Space)
        self.enc1 = nn.Linear(input_dim, 72)
        self.bn1 = nn.BatchNorm1d(72)
        self.enc2 = nn.Linear(72, 36)

        # Decoder mapping: 36 -> 72 -> Input (Reconstruction)
        self.dec1 = nn.Linear(36, 72)
        self.bn2 = nn.BatchNorm1d(72)
        self.dec2 = nn.Linear(72, input_dim)

    def forward(self, x):
        """
        Forward pass generating both reconstructions and latent codes.

        Returns:
            tuple: (Reconstructed inputs scaled by sigmoid, Latent vector).
        """
        x = F.celu(self.bn1(self.enc1(x)))
        latent = F.relu(self.enc2(x))
        x = F.celu(self.bn2(self.dec1(latent)))
        return torch.sigmoid(self.dec2(x)), latent

    def fd(self, x):
        """
        Feature extraction method used during the inference stage.
        Bypasses the decoder entirely to directly yield the latent code representation.
        """
        x = F.celu(self.bn1(self.enc1(x)))
        return F.relu(self.enc2(x))

# --- Global State Dictionary ---
# Maintains scaler states and neural network weights between Train and Test pipeline calls
# to prevent data leakage and ensure transform consistency.
# Note: The numerical values here are placeholders; actual values MUST be provided via params.json
_ae_state = {
    'model': None,
    'input_scaler': MinMaxScaler(),
    'feature_scaler': StandardScaler(),
    'device': None,          # Must be provided via "device" in params.json
    'epochs': None,          # Must be provided via "ae_epochs" in params.json
    'batch_size': None,      # Must be provided via "ae_batch_size" in params.json
    'ae_lr': None,           # Must be provided via "ae_lr" in params.json
    'ae_reg_param': None,    # Must be provided via "ae_reg_param" in params.json
    'load_model': None,      # Must be provided via "ae_load_model" in params.json
    'model_path': None       # Must be provided via "ae_model_path" in params.json
}

def set_ae_args(parameters):
    """
    Updates the global autoencoder configuration state dictionary based on
    settings parsed from the JSON file. Will raise KeyError if keys are missing.

    Args:
        parameters (dict): The configuration dictionary loaded in main.py.
    """
    # Values MUST be explicitly provided in params.json
    _ae_state['device'] = parameters['device']
    _ae_state['epochs'] = parameters['ae_epochs']
    _ae_state['batch_size'] = parameters['ae_batch_size']
    _ae_state['ae_lr'] = parameters['ae_lr']
    _ae_state['ae_reg_param'] = parameters['ae_reg_param']
    _ae_state['load_model'] = parameters['ae_load_model']
    _ae_state['model_path'] = parameters['ae_model_path']

def encode_decode(data):
    """
    Encodes raw dataset features using the Sparse Autoencoder.

    This function manages the entire PyTorch training loop dynamically. If called
    first (during Stage 1/Train), it scales data and either loads or trains the network.
    If called second (during Stage 5/Test), it strictly enforces inference logic using
    preserved scalers and models.

    Args:
        data (tuple): A tuple containing (X_raw_features, y_labels).

    Returns:
        tuple: (X_latent_encoded, y_labels).
    """
    X_raw, y = data
    is_train = _ae_state['model'] is None
    device = _ae_state['device']

    # Min-Max scale inputs appropriately prior to network ingestion
    if is_train:
        X_scaled = _ae_state['input_scaler'].fit_transform(X_raw)
    else:
        X_scaled = _ae_state['input_scaler'].transform(X_raw)

    X_tensor = torch.from_numpy(X_scaled).float().to(device)

    # --- Phase: Network Initialization or Training ---
    if is_train:
        input_dim = X_raw.shape[1]
        model = SparseAutoencoder(input_dim=input_dim).to(device)
        save_path = _ae_state['model_path']

        if _ae_state['load_model']:
            if os.path.exists(save_path):
                print(f"Loading pre-trained AutoEncoder model from {save_path}...")
                model.load_state_dict(torch.load(save_path, map_location=device))
            else:
                raise FileNotFoundError(f"JSON config requested 'ae_load_model': true, but no model was found at {save_path}!")
        else:
            print(f"Training AutoEncoder from scratch for {_ae_state['epochs']} epochs (Batch Size: {_ae_state['batch_size']})...")
            optimizer = optim.Adam(model.parameters(), lr=_ae_state['ae_lr'])
            criterion = nn.MSELoss()

            # Construct PyTorch DataLoaders for reliable Mini-Batch training
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=_ae_state['batch_size'], shuffle=True)

            model.train()
            for epoch in tqdm(range(_ae_state['epochs']), desc="AE Training", unit="epoch"):
                for batch in loader:
                    x_batch = batch[0]
                    optimizer.zero_grad()
                    outputs, latent = model(x_batch)

                    # Total Loss = MSE (Reconstruction) + L1 Penalty (Sparsity)
                    loss = criterion(outputs, x_batch) + _ae_state['ae_reg_param'] * torch.mean(torch.abs(latent))
                    loss.backward()
                    optimizer.step()

            # Persist model weights locally
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved successfully to {save_path}")

        _ae_state['model'] = model

    # --- Phase: Latent Feature Extraction (Inference) ---
    _ae_state['model'].eval()
    with torch.no_grad():
        dataset = TensorDataset(X_tensor)
        # Employ a larger, fixed batch size to optimize VRAM usage during mass inference
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        latent_list = []

        for batch in loader:
            x_batch = batch[0]
            latent_list.append(_ae_state['model'].fd(x_batch).cpu().numpy())

        X_latent = np.vstack(latent_list)

    # Standardize the extracted latent codes prior to routing them into the SOM
    if is_train:
        X_encoded = _ae_state['feature_scaler'].fit_transform(X_latent)
    else:
        X_encoded = _ae_state['feature_scaler'].transform(X_latent)

    return X_encoded, y
