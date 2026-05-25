# HybridFuzz for BipedalWalker

`hybridfuzz` 是对 BipedalWalker 中五种测试方法的组合框架：

- CureFuzz
- MDPFuzz
- QDFuzz
- G-Model
- SeqFuzz

它的目标不是简单地把五个方法各跑一遍再合并结果，而是在一个统一 fuzzing loop 中使用：

- shared seed pool
- adaptive scheduler
- unified execution/oracle
- per-method feedback signals

因此，`run_hybrid.py` 是真正的 integrated hybrid fuzzer；`run_independent_ensemble.py` 只是独立 ensemble baseline。

## Current Design

### Input

五个方法在 BipedalWalker 上的 fuzz input 都对齐为一个 15 维离散数组：

```text
shape: (15,)
value range: integers in [1, 3]
usage: env.reset(input)
```

HybridFuzz 继续使用这个输入空间。大多数方法在 `BipedalWalkerHardcore-v3` 上执行；QDFuzz 保留自己的 `BipedalWalkerHardcore-v4` 执行环境，因为 QDFuzz 需要读取重写环境中的 behavior features。

### Oracle

HybridFuzz 现在把故障判定拆成三层：

```text
did_physical_crash = done/game_over/last_reward == -100
is_reward_fault    = not did_physical_crash and total_reward < reward_fault_threshold
is_fault           = did_physical_crash or is_reward_fault
```

默认 `reward_fault_threshold = 10.0`。

日志中会同时保存 physical crash、reward fault 和 total fault，避免把真实摔倒和低 reward 性能故障混成一个不可区分的 crash 数。

### Guidance Signals

各方法在 adapter 中保留或近似使用自己的 guidance 信号：

- CureFuzz: RND intrinsic reward / uncertainty.
- MDPFuzz: fault signal and reward-drop feedback.
- QDFuzz: QD behavior descriptor and grid-cell diversity.
- G-Model: diffusion-based generation and history-distance novelty score.
- SeqFuzz: TapNet/Siamese trajectory feedback as novelty-like signal.

上层 scheduler 根据统一 reward 选择下一轮运行哪个 strategy；每个 adapter 仍可维护自己的内部状态。

## File Structure

```text
hybridfuzz/
    __init__.py
    config.py
    execution.py                  # Unified execution and oracle layer
    run_hybrid.py                 # Integrated hybrid fuzzer entry point
    run_independent_ensemble.py   # Independent ensemble baseline
    scheduler.py                  # UCB / epsilon-greedy scheduler
    shared_seed_pool.py           # Shared seed corpus and scoring
    strategy_base.py              # Base adapter interface
    adapters/
        __init__.py
        curefuzz_adapter.py
        mdpfuzz_adapter.py
        qdfuzz_adapter.py
        g_model_adapter.py
        seqfuzz_adapter.py
    utils/
        __init__.py
        crash_utils.py
        feature_extractor.py
        result_logger.py
```

## How to Run

Run commands from the `BipedalWalker` project root:

```powershell
cd D:\code\fuzzers\BipedalWalker
```

### Smoke Test

Use a small budget first to check that the environment, model, adapters, and output path work:

```powershell
python hybridfuzz\run_hybrid.py --budget 20 --scheduler ucb --output hybridfuzz\results\smoke_test
```

After it finishes, check:

```text
hybridfuzz/results/smoke_test/iteration_log.csv
hybridfuzz/results/smoke_test/summary.json
```

### Full Hybrid Run

```powershell
python hybridfuzz\run_hybrid.py --budget 10000 --scheduler ucb --output hybridfuzz\results\hybrid_ucb_seed0
```

Alternative scheduler:

```powershell
python hybridfuzz\run_hybrid.py --budget 10000 --scheduler epsilon_greedy --output hybridfuzz\results\hybrid_eps_seed0
```

### Independent Ensemble Baseline

This runs each adapter independently and merges unique fault signatures at the end. It is a baseline, not the true hybrid combine:

```powershell
python hybridfuzz\run_independent_ensemble.py --budget-per-method 2000
```

## Important Arguments

```text
--budget
    Total number of hybrid fuzzing iterations.

--scheduler
    Strategy selection mode: ucb or epsilon_greedy.

--reward-fault-threshold
    Reward threshold for performance failure. Default: 10.0.

--output
    Output directory. Default: hybridfuzz/results/

--alpha
    Weight for fault/crash score in the shared seed pool.

--beta
    Weight for novelty score.

--gamma
    Weight for diversity/QD score.

--delta
    Weight for CureFuzz uncertainty/RND score.

--eta
    Weight for G-Model score.

--lambda_cost
    Execution-cost penalty.
```

Example with explicit weights:

```powershell
python hybridfuzz\run_hybrid.py --budget 10000 --scheduler ucb --alpha 1 --beta 1 --gamma 1 --delta 1 --eta 1 --lambda_cost 0.1
```

## Outputs

Results are written to `hybridfuzz/results/` by default, or to the directory passed by `--output`.

### `config.json`

Stores the run configuration.

### `iteration_log.csv`

Per-iteration record. Important columns:

```text
iteration
selected_strategy
seed_id
candidate_id
is_fault
did_physical_crash
is_reward_fault
is_unique_crash
crash_signature
novelty_score
diversity_score
uncertainty_score
g_model_score
execution_cost
reward
seed_pool_size
survival_steps
```

### `summary.json`

High-level run summary:

```text
total_faults
physical_crashes
reward_faults
unique_crashes
time_to_first_crash
final_seed_pool_size
total_execution_time
per_strategy_selected_count
per_strategy_average_reward
```

### `strategy_stats.csv`

How often each strategy was selected and its average scheduler reward.

### `seed_pool_stats.csv`

Shared seed pool size and cumulative fault count over time.

## Notes

- QDFuzz intentionally uses `BipedalWalkerHardcore-v4` in the hybrid executor, while the other strategies use the default v3 environment.
- If QDFuzz v4 cannot be created, HybridFuzz falls back to the default environment and prints a warning.
- The current combine keeps the shared 15-dimensional input space aligned, while logging oracle results in a more fine-grained way than a single crash flag.
