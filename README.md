# Mixture-of-Experts (MoE) Compression & HPC Acceleration

**Principal Investigator:** Dr. Rabab Alomairy  
**Lead Developer:** Aryan Dutt

---

## Overview

This repository implements a high-performance, modular pipeline for compressing Mixture-of-Experts (MoE) layers in large language models (LLMs). Our goal is to reproduce, validate, and extend recent state-of-the-art methods (e.g., SVD- and CUR-based expert compression) while leveraging advanced HPC/GPU acceleration for real-world deployment.

- **Key features:**
    - Modular MoE gating, activation stats, and sensitivity analysis
    - SVD-based and (soon) CUR-based expert compression
    - GPU-native implementation, scalable to multi-GPU clusters (A100/H100 ready)
    - Flexible similarity metrics (CKA, cosine) for matrix sharing experiments
    - Comprehensive benchmarks (CPU vs GPU) and ablation studies

---

## Project Structure

```text
src/
    Core modular scripts for gating, activation, sensitivity, compression, similarity, and orchestration.
notebooks/
    Interactive walkthroughs, demo runs, benchmarking, and experiment tracking.
tests/
    Unit and integration tests for all major modules.
data/
    Synthetic weights/matrices and (optionally) real model layers for reproducibility.
results/
    Figures, logs, and output samples for rapid validation and reporting.