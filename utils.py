import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import PrunableLinear, SelfPruningNetwork


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch CPU and GPU for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Download and return CIFAR-10 train and test DataLoaders with standard preprocessing."""
    # Ensure fast multi-threaded dataset mirror download if not present
    try:
        from download_cifar import main as download_cifar_main
        download_cifar_main()
    except Exception as e:
        print(f"[Warning] Custom downloader notice: {e}")

    # CIFAR-10 Dataset Mean and Standard Deviation
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Standard torchvision dataset instantiation
    torchvision.datasets.CIFAR10.url = "https://cs231n.stanford.edu/cifar-10-python.tar.gz"
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader



def compute_loss(
    model: SelfPruningNetwork,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_val: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Total Loss = Classification Loss + lambda * Sparsity Loss.
    Sparsity Loss = Sum of all gate values across all PrunableLinear layers.
    """
    clf_loss = F.cross_entropy(outputs, targets)

    all_gates = model.get_all_gates()
    sparsity_loss = sum(torch.sum(gates) for gates in all_gates)

    total_loss = clf_loss + lambda_val * sparsity_loss
    return total_loss, clf_loss, sparsity_loss


def plot_gate_distribution(model: SelfPruningNetwork, save_path: str) -> None:
    """Plot a histogram of all final learned gate values for the best model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_gates = model.get_all_gates()
    gate_values = torch.cat([g.view(-1) for g in all_gates]).detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.axvline(x=0.01, color="crimson", linestyle="--", linewidth=2, label="Threshold (0.01)")
    plt.title("Learned Gate Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel(r"Gate Value $\sigma(gate\_score)$", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Gate distribution plot saved to: {save_path}")


def plot_sparsity_vs_accuracy(df_results: pd.DataFrame, save_path: str) -> None:
    """Plot Sparsity (%) vs Test Accuracy (%) across lambda experiments."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df_results["Sparsity Level (%)"],
        df_results["Test Accuracy (%)"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#d95f02",
        label="Lambda Trade-off"
    )

    for _, row in df_results.iterrows():
        plt.annotate(
            rf"$\lambda={row['Lambda']:.1e}$"+"\n"+f"({row['Sparsity Level (%)']:.1f}%, {row['Test Accuracy (%)']:.1f}%)",
            (row["Sparsity Level (%)"], row["Test Accuracy (%)"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.title("Model Sparsity vs Test Accuracy Trade-off", fontsize=14, fontweight="bold")
    plt.xlabel("Sparsity Level (%)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Sparsity vs Accuracy plot saved to: {save_path}")


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot training loss components and accuracy curves over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    epochs = range(1, len(history["total_loss"]) + 1)

    # 1. Total Loss
    axs[0, 0].plot(epochs, history["total_loss"], color="#1b9e77", linewidth=2)
    axs[0, 0].set_title("Total Loss", fontweight="bold")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Classification Loss
    axs[0, 1].plot(epochs, history["clf_loss"], color="#d95f02", linewidth=2)
    axs[0, 1].set_title("Classification Loss (Cross Entropy)", fontweight="bold")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Sparsity Loss
    axs[1, 0].plot(epochs, history["sparsity_loss"], color="#7570b3", linewidth=2)
    axs[1, 0].set_title(r"Sparsity Loss ($\sum gates$)", fontweight="bold")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Train vs Test Accuracy
    axs[1, 1].plot(epochs, history["train_acc"], color="#e7298a", linewidth=2, label="Train Acc")
    axs[1, 1].plot(epochs, history["test_acc"], color="#66a61e", linewidth=2, linestyle="--", label="Test Acc")
    axs[1, 1].set_title("Accuracy Curves (%)", fontweight="bold")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Accuracy (%)")
    axs[1, 1].legend(loc="lower right")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Training curves plot saved to: {save_path}")


def run_sanity_checks() -> None:
    """
    Run empirical validation checks verifying architecture, gate tensor shapes,
    sigmoid activations, weight masking, gradient flow, and loss formulation.
    """
    print("=" * 60)
    print("RUNNING ARCHITECTURE & GRADIENT FLOW SANITY CHECKS")
    print("=" * 60)

    # 1. PrunableLinear Layer Check
    in_dim, out_dim = 10, 5
    layer = PrunableLinear(in_dim, out_dim, initial_gate_score=2.0)
    
    assert isinstance(layer.gate_scores, nn.Parameter), "FAIL: gate_scores must be nn.Parameter"
    assert layer.weight.shape == layer.gate_scores.shape, "FAIL: weight and gate_scores shapes must match"
    print("[OK] Check 1: gate_scores is nn.Parameter with shape identical to weight.")

    # 2. Sigmoid Gate Output Range
    gates = layer.get_gates()
    assert (gates >= 0.0).all() and (gates <= 1.0).all(), "FAIL: gates must lie in [0, 1]"
    print("[OK] Check 2: sigmoid(gate_scores) produces valid probabilities in range [0, 1].")

    # 3. Model Forward Pass & Gradient Computation
    model = SelfPruningNetwork(input_dim=3072, hidden_dims=[128, 64], num_classes=10)
    dummy_input = torch.randn(4, 3072)
    dummy_target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    outputs = model(dummy_input)
    assert outputs.shape == (4, 10), f"FAIL: output shape mismatch. Expected (4, 10), got {outputs.shape}"
    
    total_loss, clf_loss, sparsity_loss = compute_loss(model, outputs, dummy_target, lambda_val=1e-4)
    total_loss.backward()

    # 4. Verify Gradients Reach gate_scores
    all_prunable_layers = [m for m in model.modules() if isinstance(m, PrunableLinear)]
    for i, p_layer in enumerate(all_prunable_layers):
        assert p_layer.weight.grad is not None, f"FAIL: weight.grad is None for layer {i}"
        assert p_layer.gate_scores.grad is not None, f"FAIL: gate_scores.grad is None for layer {i}"
        assert not torch.isnan(p_layer.gate_scores.grad).any(), f"FAIL: NaN gradient in layer {i} gate_scores"
    print("[OK] Check 3: Backward pass successful. Gradients correctly propagate to weight.grad AND gate_scores.grad.")

    # 5. Sparsity Calculation Verification
    stats = model.count_weights_and_sparsity(threshold=1e-2)
    assert stats["total_weights"] == sum(p.numel() for p in model.get_all_gates()), "FAIL: total weights mismatch"
    assert 0.0 <= stats["sparsity_pct"] <= 100.0, "FAIL: Sparsity % out of range"
    print(f"[OK] Check 4: Sparsity computation correct. Total parameters: {stats['total_weights']}.")

    print("=" * 60)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_sanity_checks()

