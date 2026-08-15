import math
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Custom Linear Layer with Learnable Sigmoid Gates for Dynamic Weight Pruning.
    
    Each weight matrix element W_ij has a corresponding learnable parameter
    gate_score_ij. During forward pass:
        gates = sigmoid(gate_scores)
        pruned_weights = weight * gates
        output = F.linear(x, pruned_weights, bias)
        
    Gradients flow directly to both weight and gate_scores.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_gate_score: float = 2.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. Primary trainable weights (Kaiming / He Uniform initialization)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        
        # 2. Trainable gate scores (must match weight shape exactly)
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))

        # 3. Optional bias vector
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters(initial_gate_score)

    def _reset_parameters(self, initial_gate_score: float) -> None:
        """Initialize weights using Kaiming Uniform and gate_scores to constant initial value."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize gate scores such that initial gates are reasonably active (e.g. sigmoid(2.5) ≈ 0.924)
        nn.init.constant_(self.gate_scores, initial_gate_score)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate activations in [0, 1]."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying element-wise gate multiplication before linear transformation.
        """
        # Calculate active gate values
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise soft pruning
        pruned_weights = self.weight * gates
        
        # Linear transformation with soft-pruned weights
        return F.linear(x, pruned_weights, self.bias)


class SelfPruningNetwork(nn.Module):
    """
    Feed-Forward Neural Network for CIFAR-10 with learnable self-pruning layers.
    Default architecture: 3072 -> 512 -> 256 -> 128 -> 10
    """
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 10,
        initial_gate_score: float = 2.5
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        
        # Build layers dynamically
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(PrunableLinear(current_dim, h_dim, initial_gate_score=initial_gate_score))
            layers.append(nn.ReLU())
            current_dim = h_dim
        
        # Output layer
        layers.append(PrunableLinear(current_dim, num_classes, initial_gate_score=initial_gate_score))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.network(x)

    def get_all_gates(self) -> List[torch.Tensor]:
        """Collect gate tensor outputs from all PrunableLinear layers in the model."""
        gates_list = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates_list.append(module.get_gates())
        return gates_list

    def count_weights_and_sparsity(self, threshold: float = 1e-2) -> Dict[str, float]:
        """
        Calculate weight pruning statistics based on gate values relative to threshold.
        
        Sparsity Formula:
            sparsity (%) = (number of gates < threshold / total number of gates) * 100
        """
        all_gates = self.get_all_gates()
        total_weights = 0
        pruned_weights = 0

        for gates in all_gates:
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()

        active_weights = total_weights - pruned_weights
        sparsity_pct = (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0

        return {
            "total_weights": total_weights,
            "pruned_weights": pruned_weights,
            "active_weights": active_weights,
            "sparsity_pct": sparsity_pct,
        }
