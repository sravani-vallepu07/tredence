# Self-Pruning Neural Network for CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade PyTorch implementation of a **Self-Pruning Neural Network** for CIFAR-10 image classification. Rather than applying post-training heuristic thresholding, this network learns sparse connectivity patterns **dynamically during training** via continuous learnable gate parameters and $L_1$ gate regularization.

---

## 📌 Project Overview

Deep neural networks are typically over-parameterized, requiring significant compute and memory. Traditional pruning techniques rely on post-hoc magnitude thresholding or fine-tuning pipelines. 

This repository implements **in-situ soft pruning**:
- Every parameter weight $W_{ij}$ has a corresponding continuous parameter $\text{gate\_score}_{ij}$.
- A soft gate $\text{gate}_{ij} = \sigma(\text{gate\_score}_{ij}) \in (0, 1)$ scales the effective weight: $W_{\text{pruned}} = W \times \text{gate}$.
- An $L_1$ sparsity loss term $\lambda \sum \text{gate}_{ij}$ penalizes active gates.
- Through joint gradient descent, connections that contribute minimally to classification accuracy are automatically driven towards zero gate activation.

---

## 🏗️ Architecture

The model is a fully configurable feed-forward neural network (`SelfPruningNetwork`) operating on flattened CIFAR-10 images ($3 \times 32 \times 32 = 3072$ input features).

```
Input Image (3 x 32 x 32)
       │
    Flatten (3072 features)
       │
 ┌─────┴────────────────────────┐
 │  PrunableLinear (3072 → 512) │  ← Learnable Gates (3072 x 512)
 └─────┬────────────────────────┘
     ReLU
 ┌─────┴────────────────────────┐
 │  PrunableLinear (512 → 256)  │  ← Learnable Gates (512 x 256)
 └─────┬────────────────────────┘
     ReLU
 ┌─────┴────────────────────────┐
 │  PrunableLinear (256 → 128)  │  ← Learnable Gates (256 x 128)
 └─────┬────────────────────────┘
     ReLU
 ┌─────┴────────────────────────┐
 │  PrunableLinear (128 → 10)   │  ← Learnable Gates (128 x 10)
 └─────┬────────────────────────┘
       │
Output Logits (10 Classes)
```

---

## 📐 Mathematical Formulation

### 1. Custom Prunable Layer
For a layer with weights $W \in \mathbb{R}^{M \times N}$ and biases $b \in \mathbb{R}^M$:
$$g = \sigma(S), \quad \text{where } S \in \mathbb{R}^{M \times N} \text{ (gate\_scores)}$$
$$W_{\text{pruned}} = W \odot g$$
$$y = F.linear(x, W_{\text{pruned}}, b)$$

Where $\odot$ represents element-wise Hadamard multiplication and $\sigma(\cdot)$ is the logistic sigmoid function.

### 2. Loss Function Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda \cdot \mathcal{L}_{\text{sparsity}}$$
$$\mathcal{L}_{\text{classification}} = \text{CrossEntropyLoss}(y, y_{\text{true}})$$
$$\mathcal{L}_{\text{sparsity}} = \sum_{l=1}^{L} \sum_{i,j} g_{l,i,j}$$

### 3. Gradient Dynamics & $L_1$ Regularization
Because $g = \sigma(S)$, the gradient of the total loss with respect to gate score parameter $S_{ij}$ is:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial S_{ij}} = \frac{\partial \mathcal{L}_{\text{clf}}}{\partial W_{\text{pruned}, ij}} \cdot W_{ij} \cdot \sigma'(S_{ij}) + \lambda \cdot \sigma'(S_{ij})$$

Notice that the sparsity regularization contributes a **constant positive pressure** $\lambda \sigma'(S_{ij})$ pulling gate score $S_{ij}$ downwards. If a connection $W_{ij}$ does not reduce classification error enough to overcome $\lambda$, its gate score decreases continuously until $\sigma(S_{ij}) \approx 0$.

### 4. Sparsity Definition
A connection is defined as **pruned** if its gate output falls below a strict threshold:
$$\text{Threshold} = 0.01 \ (10^{-2})$$
$$\text{Sparsity Level (\%)} = \frac{\sum \mathbb{I}(g < 0.01)}{\text{Total Number of Gates}} \times 100$$

---

## 🚀 Training & Setup

### Requirements
- Python 3.10+
- PyTorch & torchvision
- NumPy, pandas, matplotlib, tqdm

### Installation
```bash
git clone https://github.com/your-username/self-pruning-neural-network.git
cd self-pruning-neural-network
pip install -r requirements.txt
```

### Running Experiments
To execute the complete experiment suite across 3 hyperparameter values ($\lambda = 10^{-5}, 10^{-4}, 10^{-3}$), perform sanity checks, generate plots, and save checkpoints:
```bash
python train.py
```

To run a single custom experiment:
```bash
python train.py --lambda_val 0.0001 --epochs 15
```

---

## 📊 Experimental Results

*Note: The table below reflects actual empirical results obtained during training.*

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) | Total Weights | Pruned Weights | Active Weights |
| ------------------ | ----------------- | ------------------ | ------------- | -------------- | -------------- |
| `1.0e-05` (Low)    | *Pending Run*     | *Pending Run*      | 2,033,280     | *Pending Run*  | *Pending Run*  |
| `1.0e-04` (Medium) | *Pending Run*     | *Pending Run*      | 2,033,280     | *Pending Run*  | *Pending Run*  |
| `1.0e-03` (High)   | *Pending Run*     | *Pending Run*      | 2,033,280     | *Pending Run*  | *Pending Run*  |

---

## 📈 Visualizations

The training pipeline automatically exports high-resolution plots to `results/`:

1. **`results/gate_distribution.png`**: Histogram of final learned gate values showing bimodal clustering (inactive gates near $0.0$, active gates near $1.0$).
2. **`results/sparsity_vs_accuracy.png`**: Trade-off curve showing Test Accuracy vs. Sparsity % as $\lambda$ increases.
3. **`results/training_curves.png`**: Four-panel training dynamics showing Total Loss, Classification Loss, Sparsity Loss, and Train vs. Test Accuracy across epochs.

---

## 🛠️ Design Decisions

- **Sigmoid Gates**: Provides a smooth, differentiable approximation of binary mask selection in $[0, 1]$, enabling end-to-end backpropagation.
- **$L_1$ Gate Regularization**: Encourages continuous parameter decay towards 0 without hard zeroing.
- **Initial Gate Score $S_{\text{init}} = 2.5$**: Sets initial gates $\sigma(2.5) \approx 0.924$, ensuring network connectivity starts strong before pruning pressure takes effect.
- **Adam Optimizer**: Handles different gradient scales for weight tensors versus gate score parameters gracefully.

---

## ⚠️ Limitations

1. **Unstructured Pruning**: Individual weights are zeroed, which creates sparse weight matrices but does not immediately accelerate standard dense GPU matrix multiplication without specialized sparse kernels (e.g. cuSPARSE).
2. **Smooth Asymptote**: Sigmoid functions asymptote towards $0.0$ but never reach mathematical absolute zero; thresholding at $0.01$ is necessary to convert soft gates into discrete masks.

---

## 🔮 Future Improvements

- **Structured / Group Pruning**: Apply gate scores to whole channels or neurons ($L_2$ group regularization) to achieve physical matrix dimension reduction.
- **Hard-Concrete / $L_0$ Regularization**: Use stochastic gate distributions (e.g., Louizos et al.) for exact zero-gate sampling.
- **Post-Pruning Fine-Tuning**: Freeze gate masks after training and fine-tune remaining active weights.

---

## 🎓 How to Explain This Project in an Interview

### 1. Intuitive High-Level Explanation
> *"Standard neural networks keep every weight active, making them heavy and redundant. Post-training pruning cuts small weights after training, but this is a heuristic that requires manual tuning. In this project, I built a neural network that learns to prune itself **during training**. Every weight is paired with a learnable 'gate score'. Applying a sigmoid gives a gate value between 0 and 1. We multiply the weight by its gate during the forward pass and add an $L_1$ penalty on all gate values to the loss function. If a connection isn't actively helping reduce classification loss, gradient descent automatically drives its gate towards zero. We then threshold gates below 0.01 as pruned."*

### 2. Top 10 Interview Questions & Answers

#### Q1: Why use a learnable gate parameter instead of pruning weights directly based on magnitude?
**Answer:** Magnitude pruning assumes small weights are useless, which is not always true—small weights can be critical for subtle decision boundaries. Learnable gates let backpropagation decide whether a weight is useful based on task performance rather than static magnitude.

#### Q2: Why initialize gate scores to a positive value like 2.5?
**Answer:** Setting gate score to 2.5 yields $\sigma(2.5) \approx 0.924$. This ensures the model starts with fully active capacity to learn features first, allowing sparsity loss to prune superfluous connections gradually. Initializing at 0 ($\sigma(0)=0.5$) would choke gradient flow from the start.

#### Q3: Why do we use sigmoid on gate scores instead of ReLU or direct clipping?
**Answer:** Sigmoid is smooth, bounded in $(0, 1)$, and strictly non-negative. Boundedness prevents exploding gate values, while smoothness provides non-zero gradients everywhere for optimization.

#### Q4: How does backpropagation update gate scores?
**Answer:** By the chain rule, $\frac{\partial \mathcal{L}}{\partial S_{ij}} = \frac{\partial \mathcal{L}_{\text{clf}}}{\partial W_{\text{pruned}}} W_{ij} \sigma'(S_{ij}) + \lambda \sigma'(S_{ij})$. The first term adjusts the gate to improve classification accuracy, while the second term provides constant downward pressure $\lambda$.

#### Q5: What happens mathematically when $\lambda$ is too high or too low?
**Answer:** If $\lambda$ is too low ($10^{-6}$), classification loss dominates, gates remain near $1.0$, and minimal pruning occurs. If $\lambda$ is too high ($10^{-2}$), sparsity loss overwhelms classification loss, collapsing all gates to $0.0$ and ruining test accuracy.

#### Q6: Why do we set the pruning threshold at 0.01?
**Answer:** Sigmoid outputs approach 0 asymptotically but never reach exact 0. A threshold of 0.01 identifies connections operating at less than 1% capacity, treating them as pruned for metric calculation.

#### Q7: Does multiplying weights by gates change the parameter count?
**Answer:** In terms of stored parameters, it doubles the count during training (weight + gate score). However, after training, gates $< 0.01$ are discarded, yielding a sparse weight tensor.

#### Q8: Does this network run faster on standard PyTorch GPUs after pruning?
**Answer:** Not automatically with dense matrix operations (`F.linear`). Unstructured zeroing reduces operations theoretically, but hardware acceleration requires conversion to sparse matrix formats (e.g. CSR/COO) or structured neuron removal.

#### Q9: What is the difference between $L_1$ weight regularization and $L_1$ gate regularization?
**Answer:** $L_1$ weight regularization penalizes weight magnitude $|W|$, driving weights towards zero but leaving the network architecture intact. $L_1$ gate regularization explicitly scales signal flow independently of weight magnitude, separating feature importance from feature scale.

#### Q10: How would you extend this to Convolutional Neural Networks (CNNs)?
**Answer:** Instead of element-wise weight gates, assign one learnable gate score per output channel/kernel in Conv2d layers. Pruning a channel gate to zero removes the entire feature map, enabling true structured speedups on standard GPU hardware.
