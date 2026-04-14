# Deep-Gradient-Dynamics

An empirical investigation into the dynamics of gradient flow and training stability in deep neural networks.

## Overview
This repository contains a focused research notebook analyzing two of the most prominent challenges in training deep architectures: **Vanishing Gradients** and **Exploding Gradients**. 

Instead of treating neural networks as a black box, this project visualizes how error signals propagate backward through network layers and demonstrates practical, theoretically grounded techniques to mitigate gradient instability.

## Key Experiments & Analysis

### 1. Gradient Flow Visualization
We implemented a custom logging mechanism to track and visualize gradient norms across different layers of a Multi-Layer Perceptron (MLP) during training. This provides a clear, empirical look at how the signal decays (vanishes) or grows exponentially (explodes) depending on the network's depth and configuration.

### 2. The Role of Activation Functions
The notebook compares the backpropagation dynamics of different non-linearities:
* **Sigmoid:** Demonstrating the classic vanishing gradient problem due to the derivative's upper bound (0.25), which squashes the signal in deep networks.
* **ReLU:** Showing how piecewise linear functions help preserve the gradient magnitude across multiple layers.

### 3. Gradient Clipping
To address the exploding gradient problem, we explore the implementation and impact of **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_`). 

## Experimental Setup
* **Framework:** PyTorch
* **Dataset:** MNIST
* **Visualization:** Matplotlib, Seaborn (for gradient distribution histograms and flow charts)

## Structure
* The core Jupyter Notebook containing all theoretical explanations, PyTorch implementations, training loops, and visualizations.
