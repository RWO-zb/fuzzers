# HybridFuzz for BipedalWalker

This directory contains `HybridFuzz`, an integrated hybrid fuzzer that combines five different fuzzing strategies (CureFuzz, MDPFuzz, QDFuzz, G-Model, SeqFuzz) into a single, unified fuzzing loop for testing BipedalWalker environments.

**Integrated HybridFuzz = shared seed pool + adaptive scheduler + unified feedback.**
**Independent Ensemble = run original fuzzers separately and merge results.**

## File Structure

```text
hybridfuzz/
    ├── __init__.py
    ├── run_hybrid.py                 # Entry point for the integrated hybrid fuzzer
    ├── run_independent_ensemble.py   # Baseline to run methods separately and merge results
    ├── config.py                     # Hyperparameters and shared configurations
    ├── shared_seed_pool.py           # Implements the shared seed corpus and unified scoring
    ├── scheduler.py                  # Implements Epsilon-Greedy and UCB adaptive schedulers
    ├── strategy_base.py              # Abstract base class for all fuzzing adapters
    ├── adapters/
    │   ├── __init__.py
    │   ├── curefuzz_adapter.py       # Adapter for CureFuzz
    │   ├── mdpfuzz_adapter.py        # Adapter for MDPFuzz
    │   ├── qdfuzz_adapter.py         # Adapter for QDFuzz
    │   ├── g_model_adapter.py        # Adapter for G-Model
    │   └── seqfuzz_adapter.py        # Adapter for SeqFuzz
    └── utils/
        ├── __init__.py
        ├── feature_extractor.py      # Extracts unified features from environment trajectories
        ├── result_logger.py          # Handles logging of results and statistics
        └── crash_utils.py            # Helpers for crash detection and signature generation
```

## How to Run

### 1. Integrated Hybrid Fuzzer
To run the true integrated hybrid fuzzer where all strategies share the same pool and are selected adaptively:

```bash
python hybridfuzz/run_hybrid.py --budget 10000 --scheduler ucb
```

**Parameters:**
* `--env`: Environment ID (default: BipedalWalkerHardcore-v3)
* `--budget`: Total number of fuzzing iterations
* `--scheduler`: Strategy selection mode (`ucb` or `epsilon_greedy`)
* `--alpha`, `--beta`, `--gamma`, `--delta`, `--eta`: Weights for crash, novelty, diversity, uncertainty, and g-model scores.
* `--lambda_cost`: Weight for execution cost penalty.

### 2. Independent Ensemble Baseline
To run the baseline where each strategy is executed independently and the results are simply merged at the end (for experimental comparison):

```bash
python hybridfuzz/run_independent_ensemble.py --budget-per-method 2000
```
*Note: This is strictly a baseline for evaluation and does not represent the hybrid combination.*

## Adapter Implementation Details

The adapters reuse core capabilities from the existing methodologies in the parent directories. Some logic is wrapped to fit the unified `FuzzingStrategy` interface.

1. **CureFuzz Adapter**: Reuses the RND (Random Network Distillation) intrinsic reward for computing the `uncertainty_score`. Fallback mutation is used if `cure_fuzz.py` cannot be fully integrated.
2. **MDPFuzz Adapter**: Focuses on physical crash detection. Utilizes standard RL policy execution logic and standard BipedalWalker state mutation.
3. **QDFuzz Adapter**: Reuses the Quality-Diversity novelty grid (`compute_cell`, `get_edges`) to calculate the `diversity_score` based on behavior descriptors.
4. **G-Model Adapter**: Uses the `Diffusion` model for candidate generation. It periodically trains on collected test cases. Used for both generation and providing a conceptual `g_model_score`.
5. **SeqFuzz Adapter**: Reuses the `tapnet` Siamese Network to predict crashes from trajectory sequences and outputs it as a `novelty_score`. 

### Fallbacks
* If a method's complex custom mutation logic is heavily entangled in its main execution script and cannot be cleanly imported, a **fallback standard mutation** (modifying the 15-dimensional state array with a 10% chance of incrementing/decrementing values by 1 modulo 4) is applied to ensure the pipeline runs seamlessly.
* G-model handles candidate generation 50% of the time. The remaining 50% uses fallback mutation.

## Viewing Results

Results are stored in `hybridfuzz/results/` by default. 
* `summary.json`: High-level metrics like total crashes, unique crashes, and scheduler stats.
* `iteration_log.csv`: Per-iteration details of what strategy was run and the outcomes.
* `seed_pool_stats.csv`: Evolution of the shared seed pool.
* `strategy_stats.csv`: Statistics on how often each strategy was selected and their rewards.
* `ensemble_summary.json` & `ensemble_crashes.json`: Output when running the independent ensemble baseline.
