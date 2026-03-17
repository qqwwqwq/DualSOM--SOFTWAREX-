import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.style.use('ggplot')


# --- Model Definition ---
class SparseAutoencoder(nn.Module):
    # [修改] 增加 input_dim 参数，默认为 57 以兼容老代码
    def __init__(self, input_dim=57):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim

        # 将第一层和最后一层设为动态参数
        self.enc1 = nn.Linear(input_dim, 72)
        self.bn1 = nn.BatchNorm1d(72)

        self.enc2 = nn.Linear(72, 36)

        self.dec1 = nn.Linear(36, 72)
        self.bn2 = nn.BatchNorm1d(72)

        self.dec2 = nn.Linear(72, input_dim)

    def forward(self, x):
        x = F.celu(self.bn1(self.enc1(x)))
        latent = F.relu(self.enc2(x))
        x = F.celu(self.bn2(self.dec1(latent)))
        reconstructed = torch.sigmoid(self.dec2(x))

        return reconstructed, latent

    def fd(self, x):
        x = F.celu(self.bn1(self.enc1(x)))
        x = F.relu(self.enc2(x))
        return x


# --- Auxiliary Classes and Functions ---
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = []

    def add(self, value):
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
    loss = torch.mean(torch.abs(latent_code))
    return loss


# --- Core Functionality: Training Function ---
def fit(model, batch_size, tr_data, epochs, device, save_path):
    print(f'Starting training: Batch Size={batch_size}, Epochs={epochs}')
    add_sparsity = 'yes'
    reg_param = 0.001

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    tr_loss_h = History('min')

    n_sample = tr_data.shape[0]
    n_batch = (n_sample + batch_size - 1) // batch_size

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        perm = torch.randperm(n_sample)
        tr_data = tr_data[perm]

        for i in range(n_batch):
            end_idx = min((i + 1) * batch_size, n_sample)
            if end_idx - i * batch_size <= 1:
                continue

            x = tr_data[i * batch_size: end_idx].to(device)
            optimizer.zero_grad()

            outputs, latent = model(x)
            mse_loss = criterion(outputs, x)

            if add_sparsity == 'yes':
                l1_loss = sparse_loss(latent)
                loss = mse_loss + reg_param * l1_loss
            else:
                loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / n_batch
        tr_loss_h.add(epoch_loss)
        scheduler.step(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss * 10000:.4f} (Best: {tr_loss_h.best * 10000:.4f}) | LR: {current_lr:.6f}")

    print('End AutoEncoder training!')
    torch.save(model.state_dict(), save_path)
    print(f"--> Final model saved to {save_path}")

    return model


# --- Load or Train ---
def get_or_train_model(tr_data, epochs=50, batch_size=64, force_train=False, input_dim=57):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [修改] 透传 input_dim 参数
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