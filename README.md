# Replication Package for Evaluating Fuzz Testing for Reinforcement Learning Models

---

## 📌 Project Overview
This project provides a comprehensive evaluation of various reinforcement learning (RL) fuzzing approaches across the following benchmark environments:
* **MountainCar**: A foundational classic control task for evaluating exploration efficiency.
* **BipedalWalker**: A complex robotic locomotion task characterized by continuous state and action spaces.
* **CARLA**: A high-fidelity, photorealistic autonomous driving simulator for safety-critical scenario testing.

Detailed implementation and experimental configurations for each **Benchmark** across various environments can be found in the following sections:
[ **MountainCar** ](./MountainCar/) | [ **BipedalWalker** ](./BipedalWalker/) | [ **CARLA** ](./Carla/)

---

## 📁 Directory Structure

The repository is organized as follows to facilitate ease of use for reviewers and researchers:

```text
.
├── BipedalWalker/      # Data and scripts for BipedalWalker experiments
├── Carla/              # Data and scripts for CARLA experiments
├── MountainCar/        # Data and scripts for MountainCar experiments
├── plot-RQ1.py         # Script for generating Effectiveness plots (RQ1)
├── plot-RQ3.py         # Script for generating Efficiency (RQ3)
└──  plot-ven.py        # Venn diagram script for crash overlap analysis           
```

---

## 📊 Replication Guide 

Follow these steps to reproduce the Research Question (RQ) analysis presented in our study:

### RQ1: Effectiveness
```bash
python plot-RQ1.py
```
### RQ2: Diversity
```bash
python plot-ven.py
```
### RQ3: Efficiency
```bash
python plot-RQ3.py
```
### RQ4: Usefulness
The Safety monitoring task is borrowed from this awesome repository:**DRLFailureMonitor**: [https://github.com/CAgent05/DRLFailureMonitor](https://github.com/CAgent05/DRLFailureMonitor)
Run the following commands to start training.Detailed procedures for data collection are documented in the [ **BipedalWalker** ](./BipedalWalker/) section.
```bash
git clone https://github.com/CAgent05/DRLFailureMonitor.git
cd DRLFailureMonitor
python Todynet/src/train.py --dataset BipedalWalkerHCAC --nsteps 25 --epochs 100
```

---

## Supplementary Experiments

### Comparison of Input/Output Diversity Across Benchmarks

| Method | MountainCar Input Diversity | MountainCar Output Diversity | BipedalWalker Input Diversity | BipedalWalker Output Diversity | CARLA Input Diversity | CARLA Output Diversity |
|--------|-----------------------------|------------------------------|-------------------------------|--------------------------------|-----------------------|------------------------|
| Random | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 15.8 ± 15.11% | 6.6 ± 91.29% | 8.2 ± 35.97% |
| MDPFuzz | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 13 ± 9.42% | 4.4 ± 69.31% | 3.8 ± 105.92% |
| CureFuzz | 2.4 ± 37.27% | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 11.2 ± 17.17% | 7.4 ± 60.13% |
| G-Model | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 2 ± 0.00% | 10.6 ± 23.68% |
| SeqDivFuzz | 2.8 ± 39.12% | 2 ± 0.00% | 8 ± 75.52% | 2 ± 0.00% | 10.2 ± 27.20% | 8.6 ± 49.06% |
| QDFuzz | 2.4 ± 37.27% | 2 ± 0.00% | 2 ± 0.00% | 16.4 ± 12.64% | 5 ± 46.90% | 6.4 ± 68.64% |

### Comparison of TTD Across Benchmarks

TTD denotes the time to discovery, measuring how quickly each method discovers diverse inputs or outputs.

| Method | MountainCar Input TTD | MountainCar Output TTD | BipedalWalker Input TTD | BipedalWalker Output TTD | CARLA Input TTD | CARLA Output TTD |
|--------|-----------------------|------------------------|-------------------------|--------------------------|-----------------|------------------|
| Random | 0 ± 0.00% | 0 ± 0.00% | 0.0023 ± 65.72% | 0.0365 ± 13.60% | 0.6016 ± 98.49% | 0.5253 ± 55.51% |
| MDPFuzz | 0 ± 0.00% | 0 ± 0.00% | 0.0539 ± 1.45% | 0.0678 ± 9.76% | 1.9398 ± 43.61% | 1.7562 ± 51.32% |
| CureFuzz | 0.0029 ± 95.42% | 0.0042 ± 85.88% | 0.0092 ± 84.78% | 0.0135 ± 95.25% | 1.689 ± 57.78% | 1.1309 ± 82.19% |
| G-Model | 0.0113 ± 34.41% | 0.0519 ± 80.98% | 0.0057 ± 69.16% | 0.0087 ± 64.66% | 1.6595 ± 102.36% | 0.9599 ± 58.95% |
| SeqDivFuzz | 0.0325 ± 57.57% | 0.0312 ± 55.84% | 0.202 ± 90.32% | 0.0351 ± 47.30% | 3.0194 ± 32.54% | 2.0988 ± 69.74% |
| QDFuzz | 0.0048 ± 80.98% | 0.0046 ± 92.01% | 0.0022 ± 30.15% | 0.0179 ± 27.81% | 1.8846 ± 111.45% | 2.3203 ± 91.87% |

### Comparison of AUC Across Benchmarks

| Method | MountainCar Input AUC | MountainCar Output AUC | BipedalWalker Input AUC | BipedalWalker Output AUC | CARLA Input AUC | CARLA Output AUC |
|--------|-----------------------|------------------------|-------------------------|--------------------------|-----------------|------------------|
| Random | 24 ± 0.00% | 24 ± 0.00% | 23.9954 ± 0.01% | 189.0252 ± 15.12% | 72.4306 ± 87.22% | 93.5201 ± 33.12% |
| MDPFuzz | 24 ± 0.00% | 24 ± 0.00% | 23.9988 ± 0.00% | 155.8107 ± 9.41% | 46.9904 ± 61.64% | 40.0319 ± 93.24% |
| CureFuzz | 4.208 ± 37.15% | 3.505 ± 0.89% | 5.8874 ± 2.63% | 5.8787 ± 2.78% | 114.9068 ± 16.52% | 77.2839 ± 53.40% |
| G-Model | 23.9788 ± 0.04% | 23.8974 ± 0.35% | 23.9909 ± 0.03% | 23.9848 ± 0.04% | 20.728 ± 16.31% | 116.1827 ± 17.55% |
| SeqDivFuzz | 19.024 ± 38.56% | 13.6144 ± 0.46% | 16.123 ± 67.56% | 4.651 ± 3.65% | 88.9406 ± 24.40% | 82.0942 ± 48.16% |
| QDFuzz | 28.785 ± 37.27% | 23.9879 ± 0.04% | 23.9956 ± 0.01% | 196.5033 ± 12.63% | 50.9297 ± 49.88% | 60.2345 ± 64.58% |

### Finetuning Results with Regression

| Method | #UC Before Retrain | #UC After Retrain | Rob. | Acc. | Reg. |
|--------|--------------------|-------------------|------|------|------|
| Random | 4,399 | 4,095 | 6.91% | 96.00% (-2.5) | 1.00% |
| MDPFuzz | 8,952 | 6,906 | 22.86% | 99.00% (+0.5) | 1.00% |
| CureFuzz | 804 | 772 | 3.98% | 97.00% (-1.5) | 1.20% |
| G-Model | 773 | 667 | 13.71% | 95.00% (-3.5) | 1.00% |
| SeqDivFuzz | 85 | 88 | -3.53% | 98.00% (0) | 1.30% |
| QDFuzz | 2,418 | 1,483 | **38.67%** | 99.00% (+0.5) | 1.50% |

### Cross-Fuzzing-Evaluation Matrix

| Train \ Eval | Random | MDPFuzz | CureFuzz | G-Model | SeqDivFuzz | QDFuzz |
|-------------|----------|---------|---------|---------|---------|---------|
| Random | 4044 | 6763 | 802 | 712 | 98 | 1179 |
| MDPFuzz | 3927 | 7203 | 798 | 703 | 105 | 983 |
| CureFuzz | 6502 | 11291 | 804 | 672 | 100 | 1023 |
| G-Model | 5613 | 10323 | 785 | 664 | 100 | 2487 |
| SeqDivFuzz | 3629 | 8353 | 788 | 681 | 75 | 496 |
| QDFuzz | 5062 | 11702 | 792 | 622 | 104 | 1414 |

---

## 📚 Reference

Our repository is built upon or integrates the following open-source projects and research:

* **MDPFuzz-old**: [https://github.com/Qi-Pang/MDPFuzz](https://github.com/Qi-Pang/MDPFuzz)
* **MDPFuzz-new**: [https://github.com/QuentinMaz/MDPFuzz_Replicability_Study_Artifact](https://github.com/QuentinMaz/MDPFuzz_Replicability_Study_Artifact)
* **QDfuzz**: [https://github.com/QuentinMaz/QD_Based_Testing_RL](https://github.com/QuentinMaz/QD_Based_Testing_RL)
* **CureFuzz**: [https://github.com/soarsmu/CureFuzz](https://github.com/soarsmu/CureFuzz)
* **G-model**: [https://github.com/lizhuo-1994/mdp_testing](https://github.com/lizhuo-1994/mdp_testing)
* **SeqDivFuzz**: [https://github.com/AIRKEYL/SeqDivFuzz](https://github.com/AIRKEYL/SeqDivFuzz)
* **DRLFailureMonitor**: [https://github.com/CAgent05/DRLFailureMonitor](https://github.com/CAgent05/DRLFailureMonitor)
* **CARLA Simulator**: [https://github.com/carla-simulator/carla](https://github.com/carla-simulator/carla)
* **RL Baselines3 Zoo**: [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
* **PCLA**: [https://github.com/MasoudJTehrani/PCLA](https://github.com/MasoudJTehrani/PCLA)
* **Stable Baselines3 Agents**: [https://github.com/DLR-RM/rl-trained-agents](https://github.com/DLR-RM/rl-trained-agents)
