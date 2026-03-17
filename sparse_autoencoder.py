import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the plotting style to 'ggplot' for better visualization of training curves
matplotlib.style.use('ggplot')


# --- Model Definition ---
class SparseAutoencoder(nn.Module):
    """
    A Sparse Autoencoder (SAE) designed to compress input data into a 
    sparse latent representation and reconstruct it back.
    """
    def __init__(self, input_dim=57):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim

        # --- Encoder Path ---
        # First layer reduces/expands input to 72 features
        self.enc1 = nn.Linear(input_dim, 72)
        # BatchNorm helps stabilize training and allows for higher learning rates
        self.bn1 = nn.BatchNorm1d(72)
        # Bottleneck layer: creates a compressed representation (36 features)
        self.enc2 = nn.Linear(72, 36)

        # --- Decoder Path ---
        # First decoding layer attempts to reverse the bottleneck
        self.dec1 = nn.Linear(36, 72)
        self.bn2 = nn.BatchNorm1d(72)
        # Final output layer projects back to original input dimension
        self.dec2 = nn.Linear(72, input_dim)

    def forward(self, x):
        """Standard forward pass: input -> bottleneck (latent) -> reconstruction"""
        # CELU is similar to ELU but can be smoother for some datasets
        x = F.celu(self.bn1(self.enc1(x)))
        # ReLU at the bottleneck ensures latent codes are non-negative, aiding sparsity
        latent = F.relu(self.enc2(x))
        
        x = F.celu(self.bn2(self.dec1(latent)))
        # Sigmoid constrains output to [0, 1], suitable for normalized input data
        reconstructed = torch.sigmoid(self.dec2(x))

        return reconstructed, latent

    def fd(self, x):
        """Feature Detection: Returns only the latent code for downstream tasks"""
        x = F.celu(self.bn1(self.enc1(x)))
        x = F.relu(self.enc2(x))
        return x


# --- Auxiliary Classes and Functions ---
class History:
    """Utility class to track training metrics and handle early stopping logic"""
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0 # Counter for plateau detection
        self.better = False
        self.target = target
        self.history = []

    def add(self, value):
        """Updates the history with the current epoch's metric value"""
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
        self.value = value
        self.history.append(value.item() if torch.is_tensor(value) else value)


def sparse_loss(latent_code):
    """
    Calculates the L1 penalty on the latent representation.
    Encourages most neurons in the bottleneck layer to be near zero (sparse).
    """
    loss = torch.mean(torch.abs(latent_code))
    return loss


# --- Core Functionality: Training Function ---
def fit(model, batch_size, tr_data, epochs, device, save_path):
    """
    Main training loop for the Autoencoder.
    
    Args:
        model: The SparseAutoencoder instance.
        batch_size: Samples per weight update.
        tr_data: Training dataset (Tensor).
        epochs: Number of full passes through data.
        device: CPU or GPU.
        save_path: Location to store trained weights.
    """
    print(f'Starting training: Batch Size={batch_size}, Epochs={epochs}')
    add_sparsity = 'yes'
    reg_param = 0.001 # Regularization strength for sparsity

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # MSE measures how well the model reconstructs the original input
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Reduces learning rate when loss stops improving to reach finer optima
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    tr_loss_h = History('min')

    n_sample = tr_data.shape[0]
    n_batch = (n_sample + batch_size - 1) // batch_size

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Shuffle indices every epoch for stochasticity
        perm = torch.randperm(n_sample)
        tr_data = tr_data[perm]

        for i in range(n_batch):
            end_idx = min((i + 1) * batch_size, n_sample)
            # Skip batches with a single sample to avoid BatchNorm failure
            if end_idx - i * batch_size <= 1:
                continue

            x = tr_data[i * batch_size: end_idx].to(device)
            optimizer.zero_grad()

            outputs, latent = model(x)
            mse_loss = criterion(outputs, x)

            # Combined Loss = Reconstruction Loss + Sparsity Constraint
            if add_sparsity == 'yes':
                l1_loss = sparse_loss(latent)
                loss = mse_loss + reg_param * l1_loss
            else:
                loss = mse_loss

            loss.backward()
            # Gradient clipping prevents the 'exploding gradient' problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / n_batch
        tr_loss_h.add(epoch_loss)
        scheduler.step(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        # Printing loss scaled by 10k for easier monitoring of small MSE values
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss * 10000:.4f} (Best: {tr_loss_h.best * 10000:.4f}) | LR: {current_lr:.6f}")

    print('End AutoEncoder training!')
    torch.save(model.state_dict(), save_path)
    print(f"--> Final model saved to {save_path}")

    return model


# --- Load or Train ---
def get_or_train_model(tr_data, epochs=50, batch_size=64, force_train=False, input_dim=57):
    """
    Check if a pre-trained model exists; otherwise, start the training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim=input_dim)
    save_path = f'weight/sparse_ae_batch_{batch_size}.pth'

    if os.path.exists(save_path) and not force_train:
        print(f"Found saved model at {save_path}. Loading...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
    else:
        if force_train:
            print("Force train is True. Starting training...")
        else:
            print("No saved model found. Starting training...")
        model = fit(model, batch_size, tr_data, epochs, device, save_path)

    return model
