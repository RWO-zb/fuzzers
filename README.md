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
