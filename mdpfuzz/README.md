# Replication Study of MDPFuzz

This repository contains the material of the replication study conducted in the paper *Replicability Study: Policy Testing with MDPFuzz*.
As such, some of the code is borrowed from the [original implementation of MDPFuzz](https://github.com/Qi-Pang/MDPFuzz).

## Research Questions

The study aims to (RQ2) check the fault discovery ability of the fuzzers (*Fuzzer-R* and *MDPFuzz-R*) and to (RQ3) investigate the parameter sensibility of the latter (only *MDPFuzz* is parametrized).
The final version of the paper also contains a distribution analysis of the faults.
We consider the (fixed) original use cases (see the reproduction study; see `reproduction/README`) and new ones, and, compared to the reproduction study, we include in the evaluation a random testing baseline.

Besides, we increase the robustness of the results by repeating every experiment 5 times (compared to 3 conducted in the reproduction study, something we did to follow the original experimental protocol of [MDPFuzz](https://github.com/Qi-Pang/MDPFuzz)).

## Running the experiments

Each folder contains the implementation of a case study, except `rl/`, which includes the *Bipedal Walker*, *Lunar Lander* and *Taxi* use-cases.

In each sub folder, follow the `README` file which details:
1. How to manually install the Python virtual environment (only needed if you are not using the Docker image). If not, simply make sure to have the correct environment activated.
2. Execute the required computations.

The three policy testing methods studied (*Fuzzer-R*, *MDPFuzz-R*, *Random Testing*) are implemented in `methods/`.

By default, the results of the executions are exported under `data_rq2/` and `data_rq3/`.

## Additional Notes

In this study, the testing methods are run for a given number of **iterations** (5000 in total, among which 1000 are dedicated to initialize *Fuzzer-R* and *MDPFuzz-R*).
Therefore, we can't indicate the expected running times.


The parameter analysis of *MDPFuzz* (RQ3) first shows that the parameter $\tau$ has little impact on its performance (we explore 3 values: 0.01, 0.1 and 1.0), before considering a total of 20 configurations of its remaining parameters, $K$ and $\gamma$.

Therefore, for **each case study**, the parameter analysis (RQ3) requires **110** executions.
That's the reason why we definitively deem the *CARLA* case study untractable.
To that regard, we remind the user *CARLA* is compatible with Docker and that it has to be setup manually (see the instructions in `carla/`). If so, don't forget to copy the data of the container to your local system with `cp -r data_rq2/ /output`.

We separate the data analysis of RQ2 from RQ3, so that you can first execute and plot the figures for RQ2 before starting the parameter analysis (i.e, RQ3).

Similarly to the reproduction experiments (in `reproduction/`), you will find for each use case bash scripts to help you launch the experiments (namely, `launch_rq2.sh` and `launch_rq3.sh`).
To use them though, it might be required to make the scripts executable, which can be done with `chmod +x launch_rq2.sh` and `chmod +x launch_rq3.sh`.

## Computing the results and plotting the figures

To process the data and plot the results, use the virtual environment `rl` (`conda activate rl`).

### RQ2: Fault Discovery

- Compute the results of Figure 2 with `python compute_and_plot_rq2_fault.py`. The figure is saved as `rq2_fault.png` and the data in `results_rq2/`.
- Compute the results of Figure 3 with `python compute_and_plot_rq2_time.py`. The figure is saved as `rq2_time.png` and the data in `results_rq2/`.

### RQ3: Parameter Analysis

- Compute the results of Figure 5 (sensibility to $\tau$) with `python compute_and_plot_rq3_tau.py`. The figure is saved as `rq3_tau.png` and the data in `results_rq3/`.
- Compute the results of Figure 6 (sensibility to $K$ and $\gamma$) with `python compute_and_plot_rq3.py`. The figure is saved as `rq3.png` and the data in `results_rq3/`.

### Fault Distribution Analysis

Project the faults in 2D with TSNE with `python compute_and_plot_fault_analysis.py`. The figure is saved as `fault_distribution.png`.

All the `compute_*.py` scripts mentioned above account for the potential miss of data for *CARLA*.
If you are using the container, don't forget to copy the results to your local system with:
```
cp rq2_fault.png /output
cp rq2_time.png /output
cp rq3_tau.png /output
cp rq3.png /output
cp fault_distribution.png /output
```

## Data Availability

If you can't (or don't want to) reproduce all the experiments, you can download [the data we uploaded on Zenodo in the original submission](https://zenodo.org/records/10958452).

### Instructions

Download (the data using the link below) and copy-paste it in `data_rq2/`.
Then, install the virtual environment `rl`, whose instructions are detailed in `rl/`.
Finally, activate the latter and run the Python scripts mentioned above:
```bash
conda activate rl
python compute_and_plot_rq2_fault.py
python compute_and_plot_rq2_time.py
python compute_and_plot_rq3_tau.py
python compute_and_plot_rq3.py
python compute_and_plot_fault_analysis.py
```