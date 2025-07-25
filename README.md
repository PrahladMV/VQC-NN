# VQC-NN # QDigitClassify

**A hybrid quantum–classical neural network using Qiskit and PyTorch to classify handwritten digits (0–3) with a 4‑qubit variational circuit.**
---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Dependencies](#dependencies)  

---

## Overview

QDigitClassify demonstrates an end‑to‑end pipeline for classifying the first four digits of the sklearn `load_digits` dataset using a hybrid quantum neural network (QNN). Data is preprocessed with PCA, embedded into a small quantum circuit via angle‑encoding and entanglement, then refined by a classical MLP head.

---

## Features

- **Classical preprocessing:** Min–Max scaling + PCA down to 4 features  
- **Quantum feature map:** 4‑qubit angle encoding + chain entanglement + variational layer  
- **Hybrid QNN:** Qiskit’s `EstimatorQNN` wrapped as a PyTorch layer via `TorchConnector`  
- **Classical head:** Two‑layer MLP for mapping quantum expectations to digit logits  
- **Fast runtime:** Runs entirely in Google Colab (CPU or GPU) 
- **Evaluation:** Accuracy and confusion matrix plots  

---

## Architecture

1. **Data** → flatten 8×8 images → scale to [0,1] → PCA→4 features  
2. **QuantumCircuit**  
   - **Angle‑encode** each feature onto an Ry gate on its own qubit  
   - **CX** gates chain‑entangle the qubits  
   - **Variational layer** of Ry rotations (learnable parameters)  
3. **EstimatorQNN** measures 4 Pauli‑Z observables → 4 raw expectation values  
4. **PyTorch head**:  
   - `Linear(4→16) + ReLU + Linear(16→4)`  
   - `CrossEntropyLoss` + `Adam` optimizer  
5. **Train** for 20 epochs, evaluate on held‑out test set  

---

## Dependencies

```text
qiskit-aer>=0.43.1
qiskit-machine-learning>=0.9.0
torch>=1.13
torchvision
scikit-learn
matplotlib
