# 🌌 GSoC 2026: ML4SCI DeepLense — Physics-Guided Machine Learning

[![Organization](https://img.shields.io/badge/Organization-ML4SCI-blue.svg)](https://ml4sci.org/)
[![Project](https://img.shields.io/badge/Project-DeepLense-purple.svg)](https://ml4sci.org/deeplense/)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](#)

This repository contains the solution and complete data pipeline for **Specific Test VII: Physics-Guided ML**, submitted as part of the Google Summer of Code (GSoC) 2026 evaluation phase for the DeepLense project.

**Author:** Yash Yadav  
**Kaggle Profile:** [Yash2072005](https://www.kaggle.com/yash2072005)  
**LinkedIn:** [yash-yadav007](https://www.linkedin.com/in/yash-yadav007/)

---

## 📑 Project Overview

Standard "black-box" Convolutional Neural Networks often fail to generalize when applied to real-world strong gravitational lensing datasets due to observational noise and cosmic variance. The objective of this project is to develop a **Physics-Informed Neural Network (PINN)** to classify lensing images into three substructure categories:

1. **Class 0 (`no`):** No substructure (smooth dark matter halo).
2. **Class 1 (`sphere`):** Subhalo substructure (localized dark matter clumps).
3. **Class 2 (`vort`):** Vortex substructure (rotating perturbation fields).

By explicitly embedding the gravitational lensing equation directly into the network's forward pass, the model's hypothesis space is restricted to mathematically valid, physically interpretable solutions.



---

## 🔭 The Physics Foundation

This architecture analytically derives physical fields directly from a learned scalar potential map ($\psi$). 

* **Deflection Angle Field ($\alpha$):** $\alpha = \nabla \psi$
  * *Constraint:* By deriving $\alpha$ from a scalar potential, the network is guaranteed by construction to produce a curl-free (conservative) deflection field.
* **Convergence Map ($\kappa$):** $\kappa = \frac{1}{2}\nabla^2\psi$
  * *Constraint:* Represents the projected 2D mass density of the lens.
* **The Lensing Equation:** $\beta = \theta - \alpha$
  * *Constraint:* Maps the observed image plane ($\theta$) back to the unlensed source plane ($\beta$).

---

## 🧠 Architecture: The "Y-Network" PINN

To prevent gradient entanglement between the physics constraints and the classification objective, this model utilizes a decoupled "Y-Network" architecture:

1. **Physics Stem:** A lightweight convolutional block ingests the raw image and outputs the scalar lensing potential ($\psi$).
2. **Differentiable Raytracing:** Using PyTorch's `F.grid_sample`, the analytical deflection field ($\alpha$) is used to map the observed image back to the unlensed source plane ($\beta$). 
3. **Physics-Rich Fusion:** The derived spatial mass distribution ($\kappa$) is concatenated with the original image, creating a physics-enriched 4-channel input.
4. **Classification Backbone:** An `EfficientNet` backbone processes the 4-channel fusion tensor to output final class logits.

### Multi-Objective Adaptive Loss
The network is optimized using a dynamic, multi-objective loss function:
* **Classification Loss:** Standard Cross-Entropy with label smoothing.
* **Source Total Variation (TV) Loss:** Penalizes non-smoothness in the reconstructed source ($\beta$) to ensure the unlensed galaxy is morphologically plausible (compact and smooth).
* **Sparsity Regularization:** Applies a variance penalty to $\kappa$ specifically for the `sphere` class to encourage localized dark matter clumps.
* **Adaptive Weighting:** Uses dynamic lambda weighting to balance the magnitude of physics gradients against classification gradients.

---


## ⚙️ Reproducibility & Installation

The entire pipeline is contained within a single, modular Jupyter Notebook (`task-7-physics-guided-ml-rewritten.ipynb`) designed to run on Kaggle or a local GPU environment.

**Dependencies:**
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm
