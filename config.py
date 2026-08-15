import os
import torch

# Random Seed for Reproducibility
SEED = 42

# Data Configuration
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_WORKERS = 0  # 0 for safe Windows multiprocessing compatibility

# Model Architecture Configuration
INPUT_DIM = 3 * 32 * 32  # 3072 features for flattened CIFAR-10 image
HIDDEN_DIMS = [512, 256, 128]
NUM_CLASSES = 10
INITIAL_GATE_SCORE = 2.5  # sigmoid(2.5) ≈ 0.924 (initially active connections)

# Pruning & Sparsity Settings
SPARSITY_THRESHOLD = 1e-2  # Connections with gate < 0.01 are considered pruned

# Training Configuration
EPOCHS = 15
LEARNING_RATE = 1e-3
LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]  # Low, Medium, High sparsity pressures

# Directory Paths
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")
GATE_DIST_PLOT_PATH = os.path.join(RESULTS_DIR, "gate_distribution.png")
SPARSITY_VS_ACC_PLOT_PATH = os.path.join(RESULTS_DIR, "sparsity_vs_accuracy.png")
TRAINING_CURVES_PLOT_PATH = os.path.join(RESULTS_DIR, "training_curves.png")

# Hardware Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
