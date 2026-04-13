"""
PyTorch-based Sparse Autoencoder for Latent Representation Learning.

This module projects high-dimensional input features into a compact,
low-dimensional latent space (Workflow Stage 2). It respects explicit JSON
directives for either training from scratch or loading pre-trained weights,
and utilizes robust Mini-Batch training for optimal gradient convergence.
"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SparseAutoencoder(nn.Module):
    """
    Feed-forward Neural Network Architecture for the Sparse Autoencoder.

    Employs Batch Normalization to stabilize training, CELU (Continuously
    Differentiable Exponential Linear Units) for robust activation, and a
    symmetric encoder-decoder topological structure.

    Attributes:
        enc1 (nn.Linear): First encoding layer mapping input to hidden space.
        bn1 (nn.BatchNorm1d): Batch normalization for the first hidden layer.
        enc2 (nn.Linear): Second encoding layer projecting to the latent bottleneck.
        dec1 (nn.Linear): First decoding layer projecting out of the bottleneck.
        bn2 (nn.BatchNorm1d): Batch normalization for the decoder hidden layer.
        dec2 (nn.Linear): Final decoding layer reconstructing the input feature space.
    """

    def __init__(self, input_dim=57):
        """
        Initializes the autoencoder layers.

        Args:
            input_dim (int, optional): Dimensionality of the raw input feature space.
                                       Defaults to 57.
        """
        super(SparseAutoencoder, self).__init__()

        # Encoder mapping: Input -> 72 -> 36 (Latent Space Bottleneck)
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

        Args:
            x (torch.Tensor): A mini-batch of input feature vectors.

        Returns:
            tuple:
                - torch.Tensor: Reconstructed inputs scaled by a sigmoid activation [0, 1].
                - torch.Tensor: The intermediate latent vector representation.
        """
        x = F.celu(self.bn1(self.enc1(x)))
        latent = F.relu(self.enc2(x))
        x = F.celu(self.bn2(self.dec1(latent)))
        return torch.sigmoid(self.dec2(x)), latent

    def fd(self, x):
        """
        Feature extraction method used during the inference stage.

        Bypasses the decoder entirely to directly yield the latent code representation,
        saving computational overhead when reconstructions are no longer needed.

        Args:
            x (torch.Tensor): A mini-batch of input feature vectors.

        Returns:
            torch.Tensor: The encoded latent vector representation.
        """
        x = F.celu(self.bn1(self.enc1(x)))
        return F.relu(self.enc2(x))

# =====================================================================
# Global State Dictionary
# =====================================================================
# Maintains scaler states and neural network weights between Train and Test
# pipeline calls. This is critical to prevent data leakage (applying test data
# statistics to training) and ensures transform consistency.
_ae_state = {
    'model': None,
    'input_scaler': MinMaxScaler(),
    'feature_scaler': StandardScaler(),
    'device': None,          # Hardware target (e.g., 'cuda' or 'cpu')
    'epochs': None,          # Total training iterations
    'batch_size': None,      # Samples per gradient update
    'ae_lr': None,           # Optimizer learning rate
    'ae_reg_param': None,    # L1 sparsity penalty coefficient
    'load_model': None,      # Boolean flag to bypass training
    'model_path': None       # Filepath for saving/loading the .pth model
}

def set_ae_args(parameters):
    """
    Updates the global autoencoder configuration state dictionary based on
    settings parsed from the centralized JSON file.

    Args:
        parameters (dict): The configuration dictionary loaded in the main pipeline.

    Raises:
        KeyError: If any of the required AE parameters are missing from the dictionary.
    """
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

    This function dynamically manages the PyTorch execution context:
    - Train Phase (First Call): Fits the data scalers, initializes the network,
      executes the training loop (minimizing MSE + L1 Sparsity), and saves the model.
    - Inference Phase (Subsequent Calls): Freezes the network, applies the pre-fitted
      scalers to the new data, and extracts the latent representations.

    Args:
        data (tuple): A tuple containing (X_raw_features, y_labels).

    Returns:
        tuple: (X_latent_encoded, y_labels), where X_latent_encoded is a numpy array.

    Raises:
        FileNotFoundError: If `ae_load_model` is True but no weights exist at `model_path`.
    """
    X_raw, y = data

    # State tracking: If model is None, we are in the initial training/fitting phase
    is_train = _ae_state['model'] is None
    device = _ae_state['device']

    # Min-Max scale inputs to [0, 1] to match the sigmoid activation at the decoder output
    if is_train:
        X_scaled = _ae_state['input_scaler'].fit_transform(X_raw)
    else:
        X_scaled = _ae_state['input_scaler'].transform(X_raw)

    X_tensor = torch.from_numpy(X_scaled).float().to(device)

    # =========================================================
    # Phase 1: Network Initialization or Training
    # =========================================================
    if is_train:
        input_dim = X_raw.shape[1]
        model = SparseAutoencoder(input_dim=input_dim).to(device)
        save_path = _ae_state['model_path']

        if _ae_state['load_model']:
            if os.path.exists(save_path):
                print(f">>> Loading pre-trained AutoEncoder model from {save_path}...")
                model.load_state_dict(torch.load(save_path, map_location=device))
            else:
                raise FileNotFoundError(
                    f"JSON config requested 'ae_load_model': true, "
                    f"but no model was found at {save_path}!"
                )
        else:
            print(f">>> Training AutoEncoder from scratch for {_ae_state['epochs']} epochs (Batch Size: {_ae_state['batch_size']})...")
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

                    # Total Loss = MSE (Reconstruction) + L1 Penalty (Enforcing Bottleneck Sparsity)
                    loss = criterion(outputs, x_batch) + _ae_state['ae_reg_param'] * torch.mean(torch.abs(latent))
                    loss.backward()
                    optimizer.step()

            # Persist model weights locally for future bypasses
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f">>> AutoEncoder model saved successfully to {save_path}")

        # Save the initialized/trained model to the global state
        _ae_state['model'] = model

    # =========================================================
    # Phase 2: Latent Feature Extraction (Inference)
    # =========================================================
    _ae_state['model'].eval()
    with torch.no_grad():
        dataset = TensorDataset(X_tensor)
        # Employ a larger, fixed batch size to optimize VRAM usage during mass inference
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        latent_list = []

        for batch in loader:
            x_batch = batch[0]
            # Bypass decoder using fd() and move to CPU for scikit-learn compatibility
            latent_list.append(_ae_state['model'].fd(x_batch).cpu().numpy())

        X_latent = np.vstack(latent_list)

    # Standardize the extracted latent codes (Zero Mean, Unit Variance)
    # prior to routing them into the topological SOM grid
    if is_train:
        X_encoded = _ae_state['feature_scaler'].fit_transform(X_latent)
    else:
        X_encoded = _ae_state['feature_scaler'].transform(X_latent)

    return X_encoded, y
