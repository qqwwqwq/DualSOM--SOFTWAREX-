# DualSOM: Dual-mode software for clustering and classification using self-organising map

## 🕸️ Network Architecture

<p align="center">
  <img src="./assets/Outline.png" width="800">
</p>

## 📂 Data Preparation

### Tested Datasets
Our framework is highly flexible. It supports standard vectorized data (`.npy`, `.csv`) and has built-in pipelines specifically tailored and heavily evaluated on two skeleton-based human posture datasets:
* **WUT** (Warsaw University of Technology Dataset)
* **PKU-MMD** (Peking University Dataset)

### 1. Download Datasets
* **WUT Dataset**: Download the skeleton-only dataset from [Link](#).
* **PKU Dataset**: Download the skeleton-only dataset from [Link](#).

### 2. Directory Structure
After downloading and extracting, please arrange the raw data into the following directory structure before training:

```text
DualSOM/
├── Datas/
│   ├── WUT/
│   │   ├── train_data.csv       # Preprocessed training features & labels
│   │   ├── test_data.csv        # Preprocessed testing features & labels
│   │   └── wut_label_map.json   # Auto-generated label mapping
│   └── PKU/
│       ├── train_data.csv       # Preprocessed training features & labels
│       ├── test_data.csv        # Preprocessed testing features & labels
│       └── pku_label_map.json   # Auto-generated label mapping
│   └── Mnist/
│       ├── train_data.csv       # Preprocessed training features & labels
│       ├── test_data.csv        # Preprocessed testing features & labels
│       └── pku_label_map.json   # Auto-generated label mapping
├── Daulmap.py                   # Core mathematical SOM and clusterer
├── sparse_autoencoder.py        # PyTorch Mini-Batch Autoencoder module
├── preprocessing.py             # Data ingestion and generic loader
├── main.py                      # Main 5-stage execution pipeline
├── params.json                  # All-in-one configuration file
└── weight/                      # Auto-generated directory for cached models
    ├── sparse_ae.pth
    └── som_weights.npy
```

---

## ⚙️ Configuration (`params.json`)

To ensure absolute experimental reproducibility, **all hyperparameters, paths, hardware settings, and operational modes are exclusively controlled via a single `params.json` file**. 

If this file is missing, running `python main.py` will automatically generate a template.

### Configuration Template

```json
{
    "dataset_name": "wut",
    "run_mode": "supervised",
    "device": "cuda",
    "train_data_path": "Datas/WUT/train_data.csv",
    "test_data_path": "Datas/WUT/test_data.csv",
    "som_size_index": 10.0,
    "som_epochs": 50,
    "som_sigma": 4.0,
    "som_sigma_target": 0.01,
    "som_lr": 0.1,
    "som_lr_target": 0.001,
    "activation_distance": "angular",
    "som_enable_validation": 1,
    "som_load_model": false,
    "som_model_path": "weight/som_weights.npy",
    "n_clusters": 10,
    "kmeans_max_iter": 100,
    "kmeans_threshold": 0.0001,
    "ae_batch_size": 32,
    "ae_epochs": 150,
    "ae_lr": 0.001,
    "ae_reg_param": 0.001,
    "ae_load_model": false,
    "ae_model_path": "weight/sparse_ae.pth",
    "reduction_factor": 1
}
```

---

### Parameter Dictionary

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **Workflow & Paths** | | |
| `dataset_name` | String | Dataset identifier (e.g., `"wut"`, `"pku"`). Set to `"mnist"` for auto-download. |
| `run_mode` | String | `"supervised"` (classification) or `"unsupervised"` (clustering). |
| `device` | String | Hardware accelerator. Options: `"cuda"`, `"cpu"`. |
| `train_data_path` | String | Local path to the training CSV file. |
| `test_data_path` | String | Local path to the testing CSV file. |
| **Sparse Autoencoder (SAE)** | | |
| `ae_batch_size` | Int | Mini-batch size for optimal gradient convergence. |
| `ae_epochs` | Int | Total number of training epochs for the SAE. |
| `ae_lr` | Float | Learning rate for the Adam optimizer. |
| `ae_reg_param` | Float | Coefficient for the L1 sparsity penalty applied to the latent space. |
| `ae_load_model` | Bool | `true`: Load pre-trained `.pth` weights. `false`: Train from scratch. |
| `ae_model_path` | String | Filepath for saving/loading SAE weights. |
| **DualSOM** | | |
| `som_size_index` | Float | Resolution multiplier. Grid size ($N \times N$) is calculated dynamically. |
| `som_epochs` | Int | Number of epochs for topological training. |
| `som_sigma` / `lr` | Float | Initial values for neighborhood radius and learning rate. |
| `activation_distance`| String | Routing metric. Options: `"angular"`, `"euclidean"`, `"cosine"`. |
| `som_enable_validation`| Int | `1`: Prints periodic accuracy. `0`: Disables prints for max speed. *(Auto-disabled in unsupervised mode)*. |
| `som_load_model` | Bool | `true`: Load converged map weights. `false`: Train from scratch. |
| `som_model_path` | String | Filepath for saving/loading the SOM weight matrix (`.npy`). |
| **Clustering** | | |
| `n_clusters` | Int | Target number of clusters ($K$) for weight-space K-Means. |
| `kmeans_max_iter` | Int | Maximum iterations for K-Means convergence. |
| `kmeans_threshold`| Float | Convergence threshold (centroid shift) for K-Means. |

---

## 🚀 Execution and Caching

The pipeline is completely JSON-driven. When executing `python main.py`, the script sequentially processes the workflow exactly as described in the paper. Below is the core logic mapped directly to the implementation:

### Step 1: Load Parameters
The script parses command-line arguments and loads the comprehensive configuration dictionary from the JSON file, propagating settings to the Autoencoder and SOM modules.
```python
    # --- Step 1: Load parameters
    parser = argparse.ArgumentParser(description="Dual-mode SOM Pipeline with JSON config")
    parser.add_argument('--config', type=str, default='params.json', help="Path to the JSON configuration file")
    args = parser.parse_args()
    create_default_params(args.config)
    parameters = read_parameters(args.config)
```

### Step 2: Read and Encode Data 
The raw feature matrix and labels are loaded from the specified CSV files, dynamically handling structural validations. The high-dimensional features are then passed through the PyTorch-based Sparse Autoencoder, minimizing a combined MSE reconstruction loss and an L1 sparsity penalty to extract a highly condensed, low-dimensional latent code.
```python
    train_data = get_dataset(train_data_path, is_train=True, dataset_name=dataset_name)
    coded_data = encode_decode(train_data)
```

### Step 3: Create and Train DualSOM Model 
The latent codes are fed into the mathematical core. The SOM dynamically calculates its optimal grid size internally, initializes weights via PCA for rapid convergence, and adjusts its spatial topology using an exponential attention mechanism.
```python
    model = DualSOM(parameters, coded_data)
    model.fit(coded_data)
```

### Step 4: Mode-Specific Implementation (Paper Stage 4)
Depending on the `run_mode` configured in the JSON file, the framework branches into two distinct operational modes. In **Supervised Mode**, the map constructs a voting dictionary, assigning each physical neuron to the majority class of the training samples that map to it. In **Unsupervised Mode**, a specialized clusterer groups the neurons into `n_clusters`.
```python
    # --- Step 4: Mode-specific implement---
    X_train, y_train = coded_data
    # --- Step 4a: Clustering (unsupervised) ---
    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 4a: Clustering Training Phase...")
        y_pred_train = model.predict(coded_data, mode='clustering')
    # --- Step 4b: Classification (supervised, labels available) ---
    else:
        print("\n>>> Executing Stage 4b: Classification Training Phase...")
        y_pred_train = model.predict(coded_data, mode='classification')
```

### Testing Phase: Prediction on New Data 
The held-out testing data is ingested, transformed using the *frozen* Autoencoder and preserved scalers, and finally projected onto the converged SOM grid to retrieve the predicted classes or cluster IDs.
```python
    # --- Step 1: Read and encode test data ---
    test_data = get_dataset(test_data_path, is_train=False, dataset_name=dataset_name)
    coded_test = encode_decode(test_data)
    X_test, y_test = coded_test
     # --- Step 2a: Clustering (unsupervised) ---
    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 5a: Clustering Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='clustering')
        # use the same encoder as in training
     # --- Step 2b: Classification (supervised, labels available) ---
    else:
        print("\n>>> Executing Stage 5b: Classification Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='classification')
```

### 1. Standard Training
Configure your dataset paths in `params.json`, ensure `ae_load_model` and `som_load_model` are set to `false`, and run:

```bash
python main.py
```

### 2. Instant Re-evaluation (Using Cached Models)
Our framework explicitly separates training from inference. After the first run, model weights are saved to the `weight/` directory. To rapidly test a different scenario (e.g., switching to `"unsupervised"` mode with `n_clusters: (desired number of clusters)`):

1. Set `"run_mode": "unsupervised"` and `"n_clusters": (desired number of clusters)` in `params.json`.
2. Set `"ae_load_model": true` and `"som_load_model": true`.
3. Run `python main.py`.

*The pipeline will bypass training bars and output clustering metrics in seconds.*

---

## 🎯 Generic Dataset Example: MNIST

The `preprocessing.py` module includes an automated intercept for the MNIST dataset for zero-setup benchmarking.

1. Set `"dataset_name": "mnist"` in `params.json`.
2. Increase `"ae_batch_size"` to `256` and set `"n_clusters"` to `10`.
3. Run `python main.py`.

*The framework will download MNIST, normalize the 784-dim pixels, and project them onto the DualSOM grid automatically.*

---

## 📜 Reference
If you find our work useful, please consider citing:

```bibtex
@ARTICLE{10608412,
  author={He, Xin and Zielinska, Teresa and Dutta, Vibekananda and Matsumaru, Takafumi and Sitnik, Robert},
  journal={IEEE Robotics and Automation Letters}, 
  title={From Seeing to Recognising–An Extended Self-Organizing Map for Human Postures Identification}, 
  year={2024},
  volume={9},
  number={9},
  pages={7899-7906},
  doi={10.1109/LRA.2024.3433201}
}
```
