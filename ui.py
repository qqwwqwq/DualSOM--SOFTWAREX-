"""
Dual-SOM Execution Pipeline GUI
===============================
A PyQt5-based Graphical User Interface for configuring and executing a Dual Self-Organizing Map (SOM) 
and Sparse Autoencoder (SAE) machine learning pipeline. 

Features:
- Thread-safe execution of background machine learning scripts (`main.py`).
- Real-time `stdout` redirection to a graphical console.
- Interactive progress bar driven by parsed console outputs.
- Configuration persistence via `params.json`.
- Strict input validation before execution.
"""

import sys
import json
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QLineEdit, 
                             QComboBox, QCheckBox, QSpinBox, QPushButton, 
                             QLabel, QFileDialog, QGroupBox, QMessageBox, 
                             QProgressBar, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

# =====================================================================
# Default Configuration Dictionary
# =====================================================================
# Fallback parameters used when params.json is missing or corrupted.
USER_CONFIG_DEFAULTS = {
    "dataset_name": "mnist",
    "run_mode": "supervised",
    "device": "cuda",
    "train_data_path": "Datas/MNIST/train_data.csv",
    "test_data_path": "Datas/MNIST/test_data.csv",
    "som_load_model": False,
    "som_model_path": "weight/som_weights.npy",
    "ae_load_model": False,
    "ae_model_path": "weight/sparse_ae.pth",
    "auto_find_clusters": False,
    "k_min": 2,
    "k_max": 12,
    "n_clusters": 10,
    "ae_epochs": 150,
    "som_epochs": 50,
    "activation_distance": "angular"
}

# =====================================================================
# Background Worker Thread
# =====================================================================
class TrainingThread(QThread):
    """
    Executes the heavy ML pipeline in a background thread to prevent GUI freezing.
    Communicates with the main GUI thread using Qt Signals.
    """
    
    # Custom signals for cross-thread communication
    log_signal = pyqtSignal(str)          # Emits captured stdout text
    progress_signal = pyqtSignal(int)     # Emits progress percentage (0-100)
    finished_signal = pyqtSignal(str)     # Emits the path to the generated output image
    error_signal = pyqtSignal(str)        # Emits exception messages

    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self.is_running = True

    def run(self):
        """
        Main execution loop for the thread. Dumps current GUI parameters to a temp JSON,
        launches the external `main.py` script via subprocess, and continuously parses its output.
        """
        try:
            # 1. Save GUI parameters to a temporary JSON for the backend script to read
            config_path = "temp_gui_params.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_dict, f, indent=4)
            
            self.progress_signal.emit(5)
            self.log_signal.emit(f"[*] Saved execution configurations to {config_path}")
            self.log_signal.emit("[*] Initializing pipeline...\n")

            # 2. Launch the backend script
            # bufsize=1 and universal_newlines=True ensure line-buffered text parsing
            process = subprocess.Popen(
                ['python', 'main.py', '--config', config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 3. Stream output line-by-line
            for line in process.stdout:
                # Allow manual termination from the GUI
                if not self.is_running:
                    process.terminate()
                    break
                
                clean_line = line.strip()
                if clean_line:
                    self.log_signal.emit(clean_line)

                # 4. Progress bar driving mechanism based on specific stdout keywords
                if ">>> Auto-Evaluating" in clean_line:
                    self.progress_signal.emit(15)
                elif ">>> Executing Stage 4" in clean_line:
                    self.progress_signal.emit(35)
                elif "Metrics Summary - TRAINING" in clean_line:
                    self.progress_signal.emit(50)
                elif ">>> Executing Stage 2" in clean_line:
                    self.progress_signal.emit(65)
                elif "Metrics Summary - TESTING" in clean_line:
                    self.progress_signal.emit(80)
                elif ">>> Generating" in clean_line:
                    self.progress_signal.emit(90)
                elif ">>> All Done." in clean_line:
                    self.progress_signal.emit(100)

            # Wait for the subprocess to cleanly exit
            process.wait()

            # Handle non-zero exit codes indicating backend failure
            if process.returncode != 0 and self.is_running:
                raise RuntimeError(f"Process exited with code {process.returncode}")

            # 5. Determine the expected output image path based on execution parameters
            dataset_name = self.config_dict.get("dataset_name", "mnist")
            mode = self.config_dict.get("run_mode", "supervised")
            img_path = f"output/{dataset_name}_{mode}_testing_distribution.png"
            
            # Emit success signal
            if self.is_running:
                self.finished_signal.emit(img_path)

        except Exception as e:
            # Emit error signal to be caught and displayed by the GUI
            self.error_signal.emit(str(e))

    def stop(self):
        """Safely flags the thread to terminate the subprocess loop."""
        self.is_running = False

# =====================================================================
# Main GUI Construction
# =====================================================================
class DualSOMApp(QMainWindow):
    """
    Main application window managing layouts, user inputs, validation, 
    and subprocess lifecycle.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual-SOM Execution Pipeline")
        self.resize(1200, 850)

        # Load persisted parameters on startup
        self.init_params = self.load_initial_parameters()

        # Root layout setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Core Layout: Unified GroupBox with a two-column layout ---
        self.group_hyperparams = QGroupBox("Configuration & Hyperparameters")
        group_layout = QHBoxLayout(self.group_hyperparams)

        # Left column UI elements (Dataset & Paths)
        left_column = QWidget()
        self.setup_left_params_ui(left_column)
        
        # Right column UI elements (Algorithm Hyperparameters)
        right_column = QWidget()
        self.setup_right_params_ui(right_column)

        # Assign equal horizontal space (stretch=1) to both columns
        group_layout.addWidget(left_column, stretch=1)
        group_layout.addWidget(right_column, stretch=1)
        
        main_layout.addWidget(self.group_hyperparams, stretch=1)
        # ---------------------------------------------------------------

        # --- Control Panel (Buttons & Progress Bar) ---
        control_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.run_pipeline)
        
        self.btn_save_json = QPushButton("Save to params.json")
        self.btn_save_json.setMinimumHeight(40)
        self.btn_save_json.clicked.connect(self.save_current_to_json)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(40)
        
        control_layout.addWidget(self.btn_save_json, stretch=1)
        control_layout.addWidget(self.btn_run, stretch=2)
        control_layout.addWidget(self.progress_bar, stretch=3)
        main_layout.addLayout(control_layout)

        # --- Output Viewer (Log Console & Image Preview) ---
        self.result_splitter = QSplitter(Qt.Horizontal)
        
        # Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Consolas", 10))
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        self.log_console.setPlaceholderText("Execution logs and metrics will appear here...")
        
        # Image Preview
        self.img_label = QLabel("Visualization Preview")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #2b2b2b; color: #888888;")
        
        self.result_splitter.addWidget(self.log_console)
        self.result_splitter.addWidget(self.img_label)
        self.result_splitter.setSizes([450, 750]) # Default initial splitter ratio
        
        main_layout.addWidget(self.result_splitter, stretch=5)

    def load_initial_parameters(self):
        """
        Loads parameters from 'params.json'. If it doesn't exist or is corrupted,
        falls back to `USER_CONFIG_DEFAULTS` and creates a fresh file.
        """
        config_path = "params.json"
        params = USER_CONFIG_DEFAULTS.copy()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                    params.update(loaded_params)
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to parse params.json:\n{e}\nUsing hardcoded defaults.")
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
                
        return params

    def save_current_to_json(self):
        """Manually triggers saving current GUI states to 'params.json'."""
        params = self.get_current_params()
        try:
            with open("params.json", 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            QMessageBox.information(self, "Success", "Configurations saved to params.json successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def setup_left_params_ui(self, container):
        """Constructs the left-side form layout for Dataset and Path configurations."""
        layout = QFormLayout(container)
        # Adjust right margin to prevent visual crowding near the center division
        layout.setContentsMargins(0, 0, 10, 0) 
        
        self.input_dataset = QComboBox()
        self.input_dataset.addItems(["mnist", "wut", "pku", "other"])
        self.input_dataset.setCurrentText(self.init_params.get("dataset_name", "mnist"))
        layout.addRow("Dataset Name:", self.input_dataset)
        
        self.input_mode = QComboBox()
        self.input_mode.addItems(["supervised", "unsupervised"])
        self.input_mode.setCurrentText(self.init_params.get("run_mode", "supervised"))
        layout.addRow("Run Mode:", self.input_mode)
        
        self.input_device = QComboBox()
        self.input_device.addItems(["cuda", "cpu"])
        self.input_device.setCurrentText(self.init_params.get("device", "cuda"))
        layout.addRow("Hardware Device:", self.input_device)

        # File browsing horizontal layouts
        path_layout1 = QHBoxLayout()
        self.input_train_path = QLineEdit(self.init_params.get("train_data_path", ""))
        btn_browse1 = QPushButton("Browse...")
        btn_browse1.clicked.connect(lambda: self.browse_file(self.input_train_path))
        path_layout1.addWidget(self.input_train_path)
        path_layout1.addWidget(btn_browse1)
        layout.addRow("Train Data Path:", path_layout1)

        path_layout2 = QHBoxLayout()
        self.input_test_path = QLineEdit(self.init_params.get("test_data_path", ""))
        btn_browse2 = QPushButton("Browse...")
        btn_browse2.clicked.connect(lambda: self.browse_file(self.input_test_path))
        path_layout2.addWidget(self.input_test_path)
        path_layout2.addWidget(btn_browse2)
        layout.addRow("Test Data Path:", path_layout2)

        # Pre-trained SAE Model paths
        self.input_ae_load = QCheckBox("Load Pre-trained SAE Model")
        self.input_ae_load.setChecked(self.init_params.get("ae_load_model", False))
        layout.addRow("", self.input_ae_load)
        
        path_layout3 = QHBoxLayout()
        self.input_ae_path = QLineEdit(self.init_params.get("ae_model_path", ""))
        btn_browse3 = QPushButton("Browse...")
        btn_browse3.clicked.connect(lambda: self.browse_file(self.input_ae_path, "PyTorch Model (*.pth);;All Files (*)"))
        path_layout3.addWidget(self.input_ae_path)
        path_layout3.addWidget(btn_browse3)
        layout.addRow("SAE Model Path:", path_layout3)

        # Pre-trained SOM Model paths
        self.input_som_load = QCheckBox("Load Pre-trained SOM Model")
        self.input_som_load.setChecked(self.init_params.get("som_load_model", False))
        layout.addRow("", self.input_som_load)
        
        path_layout4 = QHBoxLayout()
        self.input_som_path = QLineEdit(self.init_params.get("som_model_path", ""))
        btn_browse4 = QPushButton("Browse...")
        btn_browse4.clicked.connect(lambda: self.browse_file(self.input_som_path, "Numpy Array (*.npy);;All Files (*)"))
        path_layout4.addWidget(self.input_som_path)
        path_layout4.addWidget(btn_browse4)
        layout.addRow("SOM Model Path:", path_layout4)

    def setup_right_params_ui(self, container):
        """Constructs the right-side form layout for ML Hyperparameters."""
        layout = QFormLayout(container)
        # Adjust left margin to space it away from the center division
        layout.setContentsMargins(10, 0, 0, 0) 

        self.input_auto_k = QCheckBox("Auto-find Clusters (Unsupervised only)")
        self.input_auto_k.setChecked(self.init_params.get("auto_find_clusters", False))
        layout.addRow("Clustering:", self.input_auto_k)

        self.input_k_min = QSpinBox()
        self.input_k_min.setRange(2, 50)
        self.input_k_min.setValue(self.init_params.get("k_min", 2))
        layout.addRow("K Min:", self.input_k_min)

        self.input_k_max = QSpinBox()
        self.input_k_max.setRange(3, 100)
        self.input_k_max.setValue(self.init_params.get("k_max", 12))
        layout.addRow("K Max:", self.input_k_max)

        self.input_n_clusters = QSpinBox()
        self.input_n_clusters.setRange(2, 100)
        self.input_n_clusters.setValue(self.init_params.get("n_clusters", 10))
        layout.addRow("Custom N Clusters:", self.input_n_clusters)

        self.input_distance = QComboBox()
        self.input_distance.addItems(["angular", "euclidean", "cosine"])
        self.input_distance.setCurrentText(self.init_params.get("activation_distance", "angular"))
        layout.addRow("Activation Distance:", self.input_distance)

        self.input_ae_epochs = QSpinBox()
        self.input_ae_epochs.setRange(1, 1000)
        self.input_ae_epochs.setValue(self.init_params.get("ae_epochs", 150))
        layout.addRow("SAE Epochs:", self.input_ae_epochs)

        self.input_som_epochs = QSpinBox()
        self.input_som_epochs.setRange(1, 1000)
        self.input_som_epochs.setValue(self.init_params.get("som_epochs", 50))
        layout.addRow("SOM Epochs:", self.input_som_epochs)

    def browse_file(self, line_edit, file_filter="CSV Files (*.csv);;All Files (*)"):
        """
        Generic file dialog wrapper.
        Updates the target QLineEdit if a file is selected.
        """
        fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if fname:
            line_edit.setText(fname)

    def get_current_params(self):
        """Extracts current values from all GUI widgets into a dictionary."""
        return {
            "dataset_name": self.input_dataset.currentText(),
            "run_mode": self.input_mode.currentText(),
            "device": self.input_device.currentText(),
            "train_data_path": self.input_train_path.text().strip(),
            "test_data_path": self.input_test_path.text().strip(),
            "ae_load_model": self.input_ae_load.isChecked(),
            "ae_model_path": self.input_ae_path.text().strip(),
            "som_load_model": self.input_som_load.isChecked(),
            "som_model_path": self.input_som_path.text().strip(),
            "auto_find_clusters": self.input_auto_k.isChecked(),
            "k_min": self.input_k_min.value(),
            "k_max": self.input_k_max.value(),
            "n_clusters": self.input_n_clusters.value(),
            "ae_epochs": self.input_ae_epochs.value(),
            "som_epochs": self.input_som_epochs.value(),
            "activation_distance": self.input_distance.currentText()
        }

    # =====================================================================
    # Input Validation Module
    # =====================================================================
    def validate_inputs(self, params):
        """
        Validates user inputs before launching the backend pipeline.
        Checks for empty paths, file existence, and logical hyperparameter constraints.

        Args:
            params (dict): The current configuration dictionary extracted from GUI.

        Returns:
            tuple: (is_valid (bool), error_message (str))
        """
        # 1. Validate Training Data
        if not params["train_data_path"]:
            return False, "Train Data Path cannot be empty."
        if not os.path.exists(params["train_data_path"]):
            return False, f"Train Data file not found:\n{params['train_data_path']}"

        # 2. Validate Testing Data
        if not params["test_data_path"]:
            return False, "Test Data Path cannot be empty."
        if not os.path.exists(params["test_data_path"]):
            return False, f"Test Data file not found:\n{params['test_data_path']}"

        # 3. Validate Pre-trained Models (Only if checkboxes are enabled)
        if params["ae_load_model"]:
            if not params["ae_model_path"]:
                return False, "SAE Model Path cannot be empty when 'Load Pre-trained SAE Model' is checked."
            if not os.path.exists(params["ae_model_path"]):
                return False, f"SAE Model file not found:\n{params['ae_model_path']}"

        if params["som_load_model"]:
            if not params["som_model_path"]:
                return False, "SOM Model Path cannot be empty when 'Load Pre-trained SOM Model' is checked."
            if not os.path.exists(params["som_model_path"]):
                return False, f"SOM Model file not found:\n{params['som_model_path']}"

        # 4. Validate Logical Constraints
        if params["auto_find_clusters"]:
            if params["k_min"] >= params["k_max"]:
                return False, "K Min must be strictly less than K Max for cluster auto-finding."

        return True, ""

    def run_pipeline(self):
        """
        Triggered when the 'Run Pipeline' button is clicked.
        Validates inputs, locks UI controls, resets progress, and spawns the background worker.
        """
        params = self.get_current_params()
        
        # --- Trigger input validation ---
        is_valid, error_msg = self.validate_inputs(params)
        if not is_valid:
            QMessageBox.critical(self, "Validation Error", error_msg)
            return # Block pipeline execution

        # Lock UI state
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Processing... Please wait")
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.img_label.clear()
        
        # Initialize and connect background thread
        self.thread = TrainingThread(params)
        self.thread.log_signal.connect(self.append_log)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.on_pipeline_finished)
        self.thread.error_signal.connect(self.on_pipeline_error)
        
        self.thread.start()

    def append_log(self, text):
        """Appends emitted stdout text to the GUI console and auto-scrolls to the bottom."""
        self.log_console.append(text)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_pipeline_finished(self, img_path):
        """
        Callback for successful pipeline completion. 
        Unlocks UI and attempts to load and scale the final output visualization.
        """
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Pipeline")
        self.progress_bar.setValue(100)
        self.log_console.append("\n[SUCCESS] Pipeline execution finished.")
        
        # Load and render the result image if it exists
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            # Scale pixmap smoothly while keeping aspect ratio bounded by the label size
            pixmap = pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(pixmap)
        else:
            self.img_label.setText(f"Image not found at:\n{img_path}")

    def on_pipeline_error(self, err_msg):
        """Callback for pipeline crashes/exceptions. Displays error modal and unlocks UI."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Pipeline")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Execution Error", f"An error occurred:\n{err_msg}")
        self.log_console.append(f"\n[ERROR] {err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Force 'Fusion' style for cross-platform visual consistency
    app.setStyle("Fusion") 
    
    window = DualSOMApp()
    window.show()
    sys.exit(app.exec_())
