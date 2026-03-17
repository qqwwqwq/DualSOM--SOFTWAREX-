# DualSOM: Dual-mode software for clustering and classification using self-organising map

## 🕸️ Network Architecture

<p align="center">
  <img src="./assets/Outline.png" width="800">
</p>

## 📂 Data Preparation

### Supported Datasets
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
│   │   ├── WUT_data_train/      # Training csv files
│   │   ├── WUT_data_test/       # Testing csv files
│   │   └── Preprocessed_data/   # Auto-generated cache
│   └── PKU/
│       ├── PKU_data_train/      # Training csv files
│       ├── PKU_data_test/       # Testing csv files
│       └── Preprocessed_data/   # Auto-generated cache
├── Daulmap.py
├── sparse_autoencoder.py
├── preprocessing.py
├── main.py
├── params.json                  # Hyperparameter Configuration
└── ...
```


---

## ⚙️ Configuration (`params.json`)

To make experiment tracking easier and command-line execution cleaner, all hyperparameters, paths, and training modes are controlled via a single `params.json` file. 

Here is a template of the configuration file and the detailed explanation of each parameter:

```json
{
  "dataset_name": "generic",
  "data_format": "npy",
  "train_path": "./data/train_data.npy",
  "train_label_path": "./data/train_labels.npy",
  "test_path": "./data/test_data.npy",
  "test_label_path": "./data/test_labels.npy",
  "output_dir": "./results",
  "reduction_factor": 1,
  "ae_batch_size": 32,
  "ae_epochs": 150,
  "device": "cuda",
  "force_train_ae": 1,
  "som_size_index": 10.0,
  "som_epochs": 50,
  "som_sigma": 4.0,
  "som_lr": 0.1,
  "som_enable_validation": 1,
  "run_mode": "supervised",
  "n_clusters": 5
}
```

### Parameter Dictionary

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **Data & Paths** | | |
| `dataset_name` | String | Dataset selector. Use `"wut"` or `"pku"` to trigger skeleton-specific preprocessing. Use `"generic"` for standard vector matrices. |
| `data_format` | String | File format for generic data. Options: `"npy"`, `"csv"`. |
| `train_path` | String | Path to the training features file or raw dataset folder. |
| `test_path` | String | Path to the testing features file or raw dataset folder. |
| `train_label_path` | String | Paths to the training label files. |
| `test_label_path` | String | Paths to the testing label files. |
| `output_dir` | String | Directory where evaluation visualizations and metric logs are saved. |
| `reduction_factor` | Int | Subsamples the dataset by this factor (e.g., `2` uses half the data). Useful for quick code debugging. |
| **Sparse Autoencoder** | | |
| `ae_batch_size` | Int | Batch size for training the autoencoder. |
| `ae_epochs` | Int | Total number of training epochs for the autoencoder. |
| `device` | String | Hardware accelerator. Options: `"cuda"`, `"cpu"`. |
| `force_train_ae` | Int | `1`: Force retrain the autoencoder from scratch. `0`: Automatically load pre-trained `.pth` weights if available. |
| **DualSOM** | | |
| `som_size_index` | Float | Controls the map resolution. The grid size is calculated dynamically based on sample size. |
| `som_epochs` | Int | Number of epochs for topological training. |
| `som_sigma` | Float | Initial neighborhood radius (spread) for the activation function. |
| `som_lr` | Float | Initial learning rate. Both learning rate and sigma decay exponentially. |
| `som_enable_validation` | Int | `1` (Standard Mode): Calculates accuracy periodically to plot training curves. `0` (Fast Mode): Disables intermediate evaluation for maximum speed. |
| **Evaluation Mode** | | |
| `run_mode` | String | `"supervised"`: Maps neurons to labels and calculates Accuracy/F1. `"unsupervised"`: Uses Algorithm 2 to regroup neurons without labels. |
| `n_clusters` | Int | The target number of clusters used exclusively in `unsupervised` mode (Algorithm 2). |

---

## 🚀 Training and Evaluation

With the JSON configuration in place, executing the entire pipeline (Data Loading $\rightarrow$ Sparse Autoencoder $\rightarrow$ DualSOM) is incredibly simple. 

### 1. Standard Execution
Ensure your `params.json` is configured correctly, then run:

```bash
python main.py --config params.json
```

### 2. Command-Line Override (Dynamic Tuning)
The framework allows **command-line arguments to seamlessly override JSON configurations**. This is highly useful for hyperparameter tuning or switching modes without editing the file.

**Scenario A: Switch Evaluation Branches**
If your JSON is set to `"supervised"`, but you want to instantly evaluate the unsupervised regrouping (Algorithm 2) into 8 clusters:

```bash
python main.py --config params.json --run_mode unsupervised --n_clusters 8
```

**Scenario B: Rapid Prototyping (Fast Mode)**
If you want to train from scratch at maximum speed (skipping periodic validation) for a quick test:

```bash
python main.py --config params.json --som_enable_validation 0 --force_train_ae 1
```

### 3. Checkpointing & Auto-Loading
To prevent redundant and time-consuming training, our framework automatically saves the trained autoencoder models into the `weight/` directory. By default (if `force_train_ae` is `0`), the script detects existing weights, **instantly loads them**, and skips the autoencoder training phase entirely.

**Instant Re-evaluation Example:**
Tweak the number of target clusters to 10 and re-evaluate instantly using the auto-loaded model:

```bash
python main.py --config params.json --force_train_ae 0 --run_mode unsupervised --n_clusters 10
```

## 🎯 Generic Dataset Example: MNIST

To demonstrate the framework's ability to handle standard generic vectorized data, here is a complete walkthrough using the MNIST dataset.

### Step 1: Prepare the Data
Run the provided `prepare_mnist.py` script to automatically download the MNIST dataset and flatten the $28 \times 28$ images into 784-dimensional generic feature vectors:

```bash
python prepare_mnist.py
```

This will generate the standard `.npy` feature and label matrices inside the `./data/mnist_npy/` directory.

### Step 2: Configure `params.json`
Point the framework to the generated `.npy` files and set the dataset name to `generic`. Because MNIST has 60,000 training samples and 10 classes, we increase the `ae_batch_size` and set `n_clusters` to 10:

```json
{
  "dataset_name": "generic",
  "data_format": "npy",
  "train_path": "./data/mnist_npy/train_data.npy",
  "train_label_path": "./data/mnist_npy/train_labels.npy",
  "test_path": "./data/mnist_npy/test_data.npy",
  "test_label_path": "./data/mnist_npy/test_labels.npy",
  "output_dir": "./results_mnist",
  "reduction_factor": 1,
  "ae_batch_size": 256,
  "ae_epochs": 50,
  "force_train_ae": 1,
  "som_epochs": 10,
  "som_enable_validation": 1,
  "run_mode": "supervised",
  "n_clusters": 10
}
```

### Step 3: Run the Pipeline
Once configured, simply execute the main script. The framework will automatically detect the 784-dim input, adapt the Sparse Autoencoder, and extract features for DualSOM clustering:

```bash
python main.py --config params.json
```

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
