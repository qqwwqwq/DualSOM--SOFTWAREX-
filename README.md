# DualSOM: Dual-mode software for clustering and classification using Self-Organising Map

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.2-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.0-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research--Ready-brightgreen?style=flat-square)

## 📖 Table of Contents

* [Introduction](#introduction)
* [Key Features](#key-features)
* [Network Architecture](#network-architecture)
* [Installation & Environment](#installation--environment)
* [Data Preparation](#data-preparation)
* [Configuration (`params.json`)](#configuration-paramsjson)
* [Execution and Caching](#execution-and-caching)
* [Optimal Cluster Selection (`Selection.py`)](#optimal-cluster-selection)
* [Benchmarking with Generic Datasets](#benchmarking-with-generic-datasets)
* [Reference](#reference)

---

## <a id="introduction"></a>✨ Introduction

**DualSOM** is an open-source software framework for unsupervised clustering and supervised classification of high-dimensional data. 

Learning-based recognition systems typically rely on large, manually annotated datasets, which limits scalability and adaptability in real-world applications. This software addresses these limitations by providing a unified framework that enables the automatic discovery of meaningful structures in high-dimensional data while allowing a seamless transition to recognition when labelled data are available.

The software combines sparse autoencoding for dimensionality reduction with a self-organising map (SOM) trained using distance-based learning. It is designed for researchers and engineers who require interpretable data representations, low computational overhead, and real-time performance.

## <a id="key-features"></a>🚀 Key Features

* **Unified Dual-Mode Pipeline:** Supports both automatic clustering and label-based classification without changes to the model structure. Both modes operate on the same trained SOM weights.
* **Unsupervised Clustering:** Includes automatic selection of the optimal number of clusters from a user-defined range using a modified K-Means algorithm.
* **Supervised Classification:** Constructs a neuron label map based on labelled training samples.
* **Latent Representation Learning:** Projects high-dimensional data into a compact latent space using a sparse autoencoder, reducing computational complexity.
* **Highly Configurable:** Users can configure grid size, latent dimensionality, learning-rate schedules, neighbourhood decay parameters, and distance metrics.

## <a id="network-architecture"></a>🕸️ Network Architecture

<p align="center">
  <img src="./assets/Outline.png" width="800">
</p>

## <a id="installation--environment"></a>🛠️ Installation & Environment

To ensure all mathematical and deep learning dependencies are correctly configured, we recommend using a virtual environment (Conda or venv).

### 1. Install Dependencies
Clone the repository and install the required packages using the provided `requirements.txt`:

```bash
# Optional: Create and activate a virtual environment
# conda create -n dualsom python=3.8
# conda activate dualsom

pip install -r requirements.txt
```

### 2. Verified Environment
This framework has been strictly verified with the following versions:
* **Deep Learning**: `torch==2.10.0`
* **Data Science**: `numpy==2.4.2`, `pandas==3.0.0`, `scipy==1.17.0`
* **Machine Learning**: `scikit-learn==1.4.2`
* **Visualization & Utilities**: `matplotlib==3.8.4`, `tqdm==4.66.4`

---

## <a id="data-preparation"></a>📂 Data Preparation

### Tested Datasets
Our framework is highly flexible. It supports standard vectorized data (`.npy`, `.csv`) and has built-in pipelines specifically tailored and evaluated on 

Two skeleton-based human posture datasets:
* **WUT** (Warsaw University of Technology Dataset)
* **PKU-MMD** (Peking University Dataset)

One image dataset:
* **MNIST** (Standard benchmark dataset of 28x28 grayscale handwritten digits)

One signal dataset:
* **FordA** (Automotive engine noise and vibration sensor time-series dataset from the UCR archive)
  
### 1. Download Datasets
* **WUT Dataset**: Download the skeleton-only dataset from [Link](#).
* **PKU Dataset**: Download the skeleton-only dataset from [Link](#).
* **MNIST Dataset**: Run `python prepare_mnist.py` to automatically download, normalize, and format the image data into compatible CSV files.
* **FordA Dataset**: Run `python prepare_ucr_forda.py` to automatically download and process the 1D sensor time-series data into compatible CSV files.

### 2. Directory Structure
After downloading and extracting, please arrange the raw data into the following directory structure before training:

```text
DualSOM/
├── Datas/
│   ├── WUT/
│   │   ├── train_data.csv       # Preprocessed training features & labels
│   │   ├── test_data.csv        # Preprocessed testing features & labels
│   │   └── wut_label_map.json   # Auto-generated label mapping
│   ├── PKU/
│   │   ├── train_data.csv       # Preprocessed training features & labels
│   │   ├── test_data.csv        # Preprocessed testing features & labels
│   │   └── pku_label_map.json   # Auto-generated label mapping
│   ├── MNIST/
│   │   ├── train_data.csv       # Preprocessed training features & labels
│   │   ├── test_data.csv        # Preprocessed testing features & labels
│   │   └── mnist_label_map.json # Auto-generated label mapping
│   └── FordA/
│       ├── train_data.csv       # Preprocessed training features & labels
│       ├── test_data.csv        # Preprocessed testing features & labels
│       └── forda_label_map.json # Auto-generated label mapping
├── Daulmap.py                   # Core mathematical SOM and clusterer
├── sparse_autoencoder.py        # PyTorch Mini-Batch Autoencoder module
├── preprocessing.py             # Data ingestion and generic loader
├── main.py                      # Main 5-stage execution pipeline
├── Selection.py                 # Tool for finding the optimal cluster number
├── params.json                  # All-in-one configuration file
└── weight/                      # Auto-generated directory for cached models
    ├── sparse_ae.pth
    └── som_weights.npy
```

---

## <a id="configuration-paramsjson"></a>⚙️ Configuration (`params.json`)

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

### Parameter Dictionary

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **Workflow & Paths** | | |
| `dataset_name` | String | Dataset identifier (e.g., `"wut"`, `"pku"`, `"mnist"`, `"forda"`).|
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

## <a id="execution-and-caching"></a>🚀 Execution and Caching

The pipeline is completely JSON-driven. When executing `python main.py`, the script sequentially processes the workflow exactly as described in the paper. Below is the core logic mapped directly to the implementation:

### Quick Start Commands

#### 1. Standard Training
Configure your dataset paths in `params.json`, ensure `ae_load_model` and `som_load_model` are set to `false`, and run:
```bash
python main.py
```

#### 2. Instant Re-evaluation (Using Cached Models)
Our framework explicitly separates training from inference. After the first run, model weights are saved to the `weight/` directory. To rapidly test a different scenario (e.g., switching to `"unsupervised"` mode):
1. Set `"run_mode": "unsupervised"` and `"n_clusters": <desired_number>` in `params.json`.
2. Set `"ae_load_model": true` and `"som_load_model": true`.
3. Run `python main.py`.

*The pipeline will bypass training blocks and output clustering metrics in seconds.*

---

## <a id="optimal-cluster-selection"></a>🔎 Optimal Cluster Selection (`Selection.py`)

When operating in **unsupervised mode**, selecting the optimal number of clusters ($K$) can be challenging. To assist with this, we provide `Selection.py`, a dedicated utility tool that mathematically determines the best $K_m$ using the Angular Distance Criterion $\Delta L(k)$ introduced in our framework.

### How It Works
`Selection.py` acts as a dry-run evaluator. It **bypasses any training** by forcibly loading the pre-trained weights from `main.py`, evaluates a user-defined range of $k$ values via spherical K-Means, and plots the absolute difference metric $\Delta L(k) = |L(k) - L(k-1)|$. The optimal cluster number minimizes this difference.

### Usage
**Prerequisite:** You must have run `main.py` at least once so that the model weights are successfully saved in the `weight/` directory.

Run the script from the terminal, specifying the minimum and maximum range of clusters you want to evaluate:
```bash
python Selection.py --k_min 2 --k_max 25
```

### Workflow Integration
1. Train your model using `main.py`.
2. Run `python Selection.py --k_min 5 --k_max 30`.
3. Check the terminal output and the generated visual plot for the **"Recommended Optimal Cluster Number (Km)"**.
4. Update the `"n_clusters"` field in your `params.json` with this recommended $K_m$.
5. Run `main.py` in `"unsupervised"` mode to get the final clustered outputs.

---

## <a id="benchmarking-with-generic-datasets"></a>🎯 Benchmarking with Generic Datasets

To evaluate the pipeline on standard benchmarks, use the provided preparation scripts to generate standardized CSV files. Once the data is prepared, the generic pipeline treats them as standard feature vectors.

### 1. MNIST (Image Data)
1. **Generate Data**: Run `python prepare_mnist.py`. This will create normalized 784-dimensional feature CSVs in `Datas/MNIST/`.
2. **Update Configuration**: In `params.json`, point the data paths to the new files:
   - `"train_data_path": "Datas/MNIST/train_data.csv"`
   - `"test_data_path": "Datas/MNIST/test_data.csv"`
3. **Run**: `python main.py`

### 2. FordA (1D Signal Data)
1. **Generate Data**: Run `python prepare_ucr_forda.py`. This will fetch and format the 500-dimensional sensor time-series into `Datas/FordA/`.
2. **Update Configuration**: In `params.json`, set the paths to the FordA CSV files:
   - `"train_data_path": "Datas/FordA/train_data.csv"`
   - `"test_data_path": "Datas/FordA/test_data.csv"`
3. **Run**: `python main.py`

*The framework will ingest these prepared CSVs, compress the high-dimensional signals/pixels through the Sparse Autoencoder, and project them onto the DualSOM grid automatically.*

---

## <a id="reference"></a>📜 Reference
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
