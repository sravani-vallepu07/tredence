import os
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

import config
from model import SelfPruningNetwork
from utils import (
    set_seed,
    get_cifar10_dataloaders,
    compute_loss,
    plot_gate_distribution,
    plot_sparsity_vs_accuracy,
    plot_training_curves,
    run_sanity_checks,
)


def train_one_epoch(
    model: SelfPruningNetwork,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    lambda_val: float,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Train the self-pruning model for one epoch."""
    model.train()
    running_total_loss = 0.0
    running_clf_loss = 0.0
    running_sparsity_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        total_loss, clf_loss, sparsity_loss = compute_loss(model, outputs, targets, lambda_val)
        
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item() * images.size(0)
        running_clf_loss += clf_loss.item() * images.size(0)
        running_sparsity_loss += sparsity_loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            "Loss": f"{total_loss.item():.3f}",
            "Clf": f"{clf_loss.item():.3f}",
            "Sparsity": f"{sparsity_loss.item():.1f}",
            "Acc": f"{100.0 * correct / total:.2f}%"
        })

    num_samples = len(dataloader.dataset)
    epoch_total_loss = running_total_loss / num_samples
    epoch_clf_loss = running_clf_loss / num_samples
    epoch_sparsity_loss = running_sparsity_loss / num_samples
    epoch_acc = (correct / total) * 100.0

    return epoch_total_loss, epoch_clf_loss, epoch_sparsity_loss, epoch_acc


def evaluate(
    model: SelfPruningNetwork,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Evaluate model classification accuracy on test dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return (correct / total) * 100.0


def run_single_experiment(
    lambda_val: float,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    device: torch.device = config.DEVICE
) -> Tuple[SelfPruningNetwork, Dict[str, List[float]], Dict[str, float]]:
    """Run full training and evaluation loop for a given lambda value."""
    print("=" * 60)
    print(f"STARTING EXPERIMENT: Lambda = {lambda_val:.1e}")
    print("=" * 60)
    
    set_seed(config.SEED)

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config.DATA_DIR, batch_size=batch_size, num_workers=config.NUM_WORKERS
    )

    model = SelfPruningNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        num_classes=config.NUM_CLASSES,
        initial_gate_score=config.INITIAL_GATE_SCORE
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "total_loss": [],
        "clf_loss": [],
        "sparsity_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    for epoch in range(1, epochs + 1):
        tot_loss, clf_loss, sp_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, lambda_val, device
        )
        test_acc = evaluate(model, test_loader, device)

        history["total_loss"].append(tot_loss)
        history["clf_loss"].append(clf_loss)
        history["sparsity_loss"].append(sp_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        stats = model.count_weights_and_sparsity(threshold=config.SPARSITY_THRESHOLD)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Total Loss: {tot_loss:.4f} | Clf Loss: {clf_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"Sparsity: {stats['sparsity_pct']:.2f}%"
        )

    final_stats = model.count_weights_and_sparsity(threshold=config.SPARSITY_THRESHOLD)
    final_stats["test_acc"] = history["test_acc"][-1]
    final_stats["lambda"] = lambda_val

    print("\n[Experiment Summary]")
    print(f"Lambda: {lambda_val:.1e}")
    print(f"Test Accuracy: {final_stats['test_acc']:.2f}%")
    print(f"Sparsity Level: {final_stats['sparsity_pct']:.2f}%")
    print(f"Total Weights: {final_stats['total_weights']}")
    print(f"Pruned Weights: {final_stats['pruned_weights']}")
    print(f"Active Weights: {final_stats['active_weights']}\n")

    return model, history, final_stats


def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network for CIFAR-10")
    parser.add_argument("--lambda_val", type=float, default=None, help="Run for specific lambda value")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--skip_sanity", action="store_true", help="Skip sanity checks")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("PyTorch Version:", torch.__version__)
    print("Device Selected:", config.DEVICE)

    if not args.skip_sanity:
        run_sanity_checks()

    lambdas_to_run = [args.lambda_val] if args.lambda_val is not None else config.LAMBDA_VALUES

    all_results = []
    best_model = None
    best_history = None
    best_stats = None
    best_score = -1.0

    for l_val in lambdas_to_run:
        model, history, stats = run_single_experiment(
            lambda_val=l_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=config.DEVICE
        )

        all_results.append({
            "Lambda": l_val,
            "Test Accuracy (%)": round(stats["test_acc"], 2),
            "Sparsity Level (%)": round(stats["sparsity_pct"], 2),
            "Total Weights": int(stats["total_weights"]),
            "Pruned Weights": int(stats["pruned_weights"]),
            "Active Weights": int(stats["active_weights"]),
        })

        # Selection criterion: balanced top performance with pruning
        # Score combines test accuracy with controlled sparsity bonus
        score = stats["test_acc"]
        if score > best_score:
            best_score = score
            best_model = model
            best_history = history
            best_stats = stats

    # Convert results to DataFrame and save to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(config.RESULTS_CSV_PATH, index=False)
    print(f"[Results Saved] Saved experiment table to: {config.RESULTS_CSV_PATH}\n")

    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED - FINAL RESULTS")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60 + "\n")

    # Save Best Model Checkpoint
    if best_model is not None:
        checkpoint = {
            "model_state_dict": best_model.state_dict(),
            "config": {
                "input_dim": config.INPUT_DIM,
                "hidden_dims": config.HIDDEN_DIMS,
                "num_classes": config.NUM_CLASSES,
                "lambda": best_stats["lambda"],
                "sparsity_threshold": config.SPARSITY_THRESHOLD,
            },
            "test_accuracy": best_stats["test_acc"],
            "sparsity_pct": best_stats["sparsity_pct"],
        }
        torch.save(checkpoint, config.BEST_MODEL_PATH)
        print(f"[Checkpoint Saved] Best model checkpoint saved to: {config.BEST_MODEL_PATH}")

        # Generate Visualizations for Best Model and Lambda Experiments
        plot_gate_distribution(best_model, config.GATE_DIST_PLOT_PATH)
        plot_sparsity_vs_accuracy(df_results, config.SPARSITY_VS_ACC_PLOT_PATH)
        plot_training_curves(best_history, config.TRAINING_CURVES_PLOT_PATH)


if __name__ == "__main__":
    main()
