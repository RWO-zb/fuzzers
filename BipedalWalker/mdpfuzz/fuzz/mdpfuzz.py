import os
import time
import copy
import json
import tqdm
import numpy as np
import torch
import pickle
import random

from typing import List, Tuple, Dict, Any

if __package__ is None or __package__ == '':
    from gmm import CoverageModel
    from logger import FuzzerLogger
    from executor import Executor
    from pool import Pool, IndexedPool, LightPool
else:
    from .gmm import CoverageModel
    from .logger import FuzzerLogger
    from .executor import Executor
    from .pool import Pool, IndexedPool, LightPool

# ==========================================
# [Alignment] TodyNet 辅助函数
# ==========================================
def process_episode_data(sequence, label, window_size):
    seq_len = len(sequence)
    if seq_len < window_size:
        return None, None
    
    seq_array = np.array(sequence) 
    windows = []
    labels = []

    if label == 0:
        max_idx = seq_len - window_size
        rand_idx = random.randint(0, max_idx)
        win = seq_array[rand_idx : rand_idx + window_size]
        win = win.transpose() 
        windows.append(win)
        labels.append(0)
    else:
        win = seq_array[-window_size:] 
        win = win.transpose() 
        windows.append(win)
        labels.append(1)
        
    return np.array(windows), np.array(labels)

def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    if not X_list:
        return
    
    print(f"\n[TodyNet Data] Processing balancing to {target_total} samples (Target Crash Ratio: {target_crash_ratio:.0%})...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    
    n_crash_target = int(target_total * target_crash_ratio)
    n_succ_target = target_total - n_crash_target
    
    if len(indices_fail) >= n_crash_target:
        final_fail = np.random.choice(indices_fail, size=n_crash_target, replace=False)
    else:
        final_fail = indices_fail
        
    if len(indices_succ) >= n_succ_target:
        final_succ = np.random.choice(indices_succ, size=n_succ_target, replace=False)
    else:
        final_succ = indices_succ
    
    final_indices = np.concatenate([final_fail, final_succ])
    np.random.shuffle(final_indices)
    
    X_balanced = X_all[final_indices]
    y_balanced = y_all[final_indices]

    X_final = np.expand_dims(X_balanced, axis=1)
    
    X_tensor = torch.from_numpy(X_final).float()
    y_tensor = torch.from_numpy(y_balanced).long()
    
    total = X_tensor.size(0)
    indices = torch.randperm(total)
    split = int(0.8 * total)
    
    ds_id = f"{dataset_name}_{window_size}"
    save_path = os.path.join(output_dir, ds_id)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(X_tensor[indices[:split]], os.path.join(save_path, 'X_train.pt'))
    torch.save(y_tensor[indices[:split]], os.path.join(save_path, 'y_train.pt'))
    torch.save(X_tensor[indices[split:]], os.path.join(save_path, 'X_valid.pt'))
    torch.save(y_tensor[indices[split:]], os.path.join(save_path, 'y_valid.pt'))
    
    print(f"[TodyNet Data] Saved {total} samples to {save_path} (Format: N,1,Feat,Win)")


class Fuzzer():
    def __init__(self, random_seed: int, k: int, tau: float, gamma: float, executor: Executor) -> None:
        self.k = k
        self.tau = tau
        self.gamma = gamma
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed) 

        self.coverage_model = CoverageModel(random_seed, k, gamma)
        self.evaluated_solutions = []
        self.mutation_history = [] 

        self.executor = executor
        self.sim_steps = self.executor.sim_steps
        self.env_seed = self.executor.env_seed

        self._set_config()
        
        self.all_window_data = [] 
        self.all_label_data = []
        self.crash_transitions = []
        self.success_transitions = []
        
        self.todynet_success_count = 0

        # [新增] 对齐评估指标 - 初始化统计变量
        self.evaluation_results = []
        self.total_env_sim_time = 0.0
        self.fuzzing_start_time = 0.0

    def _set_config(self):
        self.config = {
            'k': self.k,
            'gamma': self.gamma,
            'tau': self.tau,
            'random_seed': self.random_seed,
            'random_state': self.rng.bit_generator.state,
            'env_seed': self.env_seed,
            'sim_steps': self.sim_steps,
            'name': 'MDPFuzz',
            'use_case': type(self.executor).__name__
        }

    # [新增] 对齐评估指标 - 统一记录单次执行结果的辅助函数
    def _record_evaluation(self, state, acc_reward, done, obs_seq, generation, exec_time):
        """
        参照 enjoy_cure.py 逻辑：
        - 物理碰撞直接对应 done。
        - 奖励故障对应 (not done) 且 acc_reward < 10。
        - 轨迹数据强制转为 2D (steps, 1) 适配 test1.py。
        """
        elapsed = time.time() - self.fuzzing_start_time
        
        # 参照 enjoy_cure.py: did_crash = done
        did_crash = bool(done)
        # 参照 enjoy_cure.py: elif episode_reward < 10: is_reward_fault = True
        is_reward_fault = bool((not did_crash) and (acc_reward < 10))
        
        # 轨迹数据处理：取 obs[0] 序列并重塑为 2D 矩阵 (steps, 1)
        # 这样 test1.py 的 np.pad(t, ((0, pad_len), (0, 0))) 就能正常运行
        traj_data = None
        if did_crash or is_reward_fault:
            traj_data = np.array([o[0] for o in obs_seq], dtype=np.float32).reshape(-1, 1)

        self.evaluation_results.append({
            'mutate_state': state.copy(),
            'did_crash': did_crash,
            'is_reward_fault': is_reward_fault,
            'elapsed_time': float(elapsed),
            'survival_steps': int(len(obs_seq)),
            'parent_depth': int(generation),
            'output_trajectory': traj_data
        })
        # 累加仿真时间
        self.total_env_sim_time += exec_time

    # [新增] 对齐评估指标 - 统一保存方法
    def _finalize_evaluation_logs(self, saving_path):
        if not saving_path:
            return
        
        log_dir = os.path.dirname(saving_path)
        total_wall_time = time.time() - self.fuzzing_start_time
        
        # 1. 保存 selection_log.pkl
        sel_path = os.path.join(log_dir, 'selection_log.pkl')
        with open(sel_path, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
            
        # 2. 保存 perf_meta.pkl
        meta_path = os.path.join(log_dir, 'perf_meta.pkl')
        perf_meta = {
            'total_wall_time': float(total_wall_time),
            'env_sim_time': float(self.total_env_sim_time),
            'algo_logic_time': float(total_wall_time - self.total_env_sim_time)
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(perf_meta, f)
        
        print(f"[Evaluation] Hardcoded logs saved to {log_dir}: selection_log.pkl, perf_meta.pkl")

    def _concatenate_state_sequence(self, state_sequence: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(state_sequence) - 1):
            data_concat.append(np.hstack([state_sequence[i], state_sequence[i+1]]))
        return np.array(data_concat)


    def sampling(self, n: int = 1) -> List[np.ndarray]:
        if n == 1:
            return self.executor.generate_input(self.rng)
        else:
            return self.executor.generate_inputs(self.rng, n=n)


    def mutate(self, state: np.ndarray, **kwargs):
        return self.executor.mutate(state, self.rng, **kwargs)


    def mutate_validate(self, state: np.ndarray, **kwargs):
        attempts = 1
        while attempts < 100:
            mutate_states = self.mutate(state, **kwargs)
            tmp = mutate_states.tolist()
            if not (tmp in self.evaluated_solutions):
                self.evaluated_solutions.append(tmp)
                break
            else:
                attempts += 1
        return mutate_states


    def mdp(self, state: np.ndarray, policy: Any = None) -> Tuple[float, bool, np.ndarray, float, float, float, List]:
        episode_reward, done, obs_seq, exec_time, bd_dist, bd_angle, transitions = self.executor.execute_policy(state, policy)
        return episode_reward, done, obs_seq, exec_time, bd_dist, bd_angle, transitions


    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None, generation: int = None ,**kwargs) -> Tuple[float, float, bool, List[np.ndarray], float, float, float, List]:
        perturbed_state = self.mutate_validate(state, **kwargs)
        perturbation = np.linalg.norm(state - perturbed_state)

        bd_dist_ret = None
        bd_angle_ret = None
        transitions_ret = []

        if acc_reward is None:
            acc_reward, crash, state_sequence, exec_time, bd_dist_ret, bd_angle_ret, transitions_ret = self.mdp(state, policy)
        else:
            state_sequence = []
            crash = None
            exec_time = None
            bd_dist_ret = None
            bd_angle_ret = None
            transitions_ret = []

        acc_reward_perturbed, crash_perturbed, state_sequence_perturbed, exec_time_perturbed, bd_dist_p, bd_angle_p, transitions_p = self.mdp(perturbed_state, policy)
        
        # [新增] 对齐评估指标 - 累加环境仿真时间
        self.total_env_sim_time += exec_time_perturbed

        if self.logger is not None:
            episode_length = len(state_sequence_perturbed)
            self.logger.log(
                input=perturbed_state,
                oracle=crash_perturbed,
                reward=acc_reward_perturbed,
                episode_length=episode_length,
                Generation=generation,
                test_exec_time=exec_time_perturbed,
                run_time=time.time(),
                bd_distance=bd_dist_p,
                bd_mean_angle=bd_angle_p,
                root_id=None 
            )

        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        return sensitivity, acc_reward, crash, state_sequence, exec_time, bd_dist_ret, bd_angle_ret, transitions_ret


    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation


    def initialize_coverage_model(self, **kwargs) -> int:
        exec_counter = kwargs.get('exec_counter', 0)
        state_sequence = kwargs.pop('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            reward, crash, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.mdp(random_input, policy)
            exec_counter += 1
            
            # [新增] 对齐评估指标 - 记录初始化模型阶段
            self._record_evaluation(random_input, reward, crash, state_sequence, 0, exec_time)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=crash,
                    reward=reward,
                    episode_length=episode_length,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle,
                    root_id=-1 
                    )

        if len(state_sequence) < self.k + 1:
            kwargs['exec_counter'] = exec_counter
            return self.initialize_coverage_model(**kwargs)
        else:
            self.coverage_model.initialize(state_sequence)
        print('Coverage model initialized')
        return exec_counter

    def _collect_data(self, transitions, crash, window_size, 
                      TARGET_CRASH=10000, TARGET_SUCCESS=90000, 
                      save_data=False, save_transitions=False):
        if len(transitions) == 0:
            return

        if save_transitions:
            if crash:
                if len(self.crash_transitions) < TARGET_CRASH:
                    self.crash_transitions.extend(transitions)
            else:
                if len(self.success_transitions) < TARGET_SUCCESS:
                    self.success_transitions.extend(transitions)

        if save_data:
            TODYNET_SUCCESS_CAP = 3000 
            
            collect_this = False
            if crash:
                collect_this = True
            else:
                if self.todynet_success_count < TODYNET_SUCCESS_CAP:
                    collect_this = True
            
            if collect_this:
                todynet_seq = []
                for t in transitions:
                    s, a, _, _, _ = t
                    if isinstance(a, (int, float, np.integer, np.floating)):
                         a = np.array([a])
                    vec = np.concatenate([s, a])
                    todynet_seq.append(vec)
                
                label = 1 if crash else 0
                wins, labels = process_episode_data(todynet_seq, label, window_size)
                if wins is not None and len(wins) > 0:
                    self.all_window_data.append(wins)
                    self.all_label_data.append(labels)
                    if not crash:
                        self.todynet_success_count += 1

    # =========================================================================
    # 1. 主 MDPFuzz 循环
    # =========================================================================
    def fuzzing(self, n: int, policy: Any = None, **kwargs):
        # [新增] 对齐评估指标 - 开启计时
        self.fuzzing_start_time = time.time()

        save_data = kwargs.get('save_data', False)
        save_transitions = kwargs.get('save_transitions', False)
        window_size = kwargs.get('window_size', 20)
        
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None

        self.root_lineage = {}
        local_sensitivity = kwargs.get('local_sensitivity', False)

        initial_inputs = self.sampling(n)
        self.config['init_budget'] = n
        if kwargs.get('light_pool', False):
            pool = LightPool()
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))

        num_initial_executions = self.initialize_coverage_model(policy=policy)
        self.config['num_initial_executions'] = num_initial_executions
        pbar = tqdm.tqdm(total=n)
        
        for i, state in enumerate(initial_inputs):
            current_root_id = i
            self.root_lineage[state.tobytes()] = current_root_id

            sensitivity, acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.sentivity(state, policy=policy, generation=0, **kwargs)
            
            # [新增] 对齐评估指标 - 记录初始阶段执行结果
            self._record_evaluation(state, acc_reward, oracle, state_sequence, 0, exec_time)

            self._collect_data(transitions, oracle, window_size, save_data=save_data, save_transitions=save_transitions)
            
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            pool.add(state, acc_reward, coverage, sensitivity, oracle, generation=0)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    coverage=coverage,
                    Generation=0,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=time.time(),
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle,
                    root_id=current_root_id 
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            test_budget -=  (2 * n) + num_initial_executions
            pbar = tqdm.tqdm(total=test_budget)
            self.config['test_budget'] = test_budget
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)
            self.config['test_budget_in_seconds'] = test_budget_in_seconds
        
        try:
            while True:
                if test_budget_in_seconds is None:
                    if num_iterations == test_budget: break
                else:
                    if (current_time - start_time) > test_budget_in_seconds: break

                input_selected, acc_reward_input, generation = pool.select(self.rng)
                parent_root_id = self.root_lineage.get(input_selected.tobytes(), -1)

                new_generation = generation + 1
                mutant = self.mutate_validate(input_selected, **kwargs)
                self.root_lineage[mutant.tobytes()] = parent_root_id
                
                acc_reward_mutant, oracle, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.mdp(mutant, policy)
                
                # [新增] 对齐评估指标 - 记录变异阶段执行结果
                self._record_evaluation(mutant, acc_reward_mutant, oracle, state_sequence, new_generation, exec_time)

                self._collect_data(transitions, oracle, window_size, save_data=save_data, save_transitions=save_transitions)

                state_sequence_conc = self._concatenate_state_sequence(state_sequence)
                t0 = time.time()
                coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
                coverage_time = time.time() - t0
                sensitivity = None
                
                if oracle:
                    pool.add_crash(mutant)
                elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                    if local_sensitivity:
                        sensitivity = self.local_sensitivity(input_selected, mutant, acc_reward_input, acc_reward_mutant)
                    else:
                        sensitivity, _, _, _, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                    pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle, generation=new_generation)

                if self.logger is not None:
                    episode_length = len(state_sequence)
                    self.logger.log(
                        input=mutant,
                        oracle=oracle,
                        reward=acc_reward_mutant,
                        episode_length=episode_length,
                        sensitivity=sensitivity,
                        coverage=coverage,
                        Generation=new_generation,
                        test_exec_time=exec_time,
                        coverage_time=coverage_time,
                        run_time=time.time(),
                        bd_distance=bd_dist,
                        bd_mean_angle=bd_angle,
                        root_id=parent_root_id 
                    )

                if test_budget_in_seconds is None:
                    num_iterations += 1
                    pbar.update(1)
                else:
                    current_time = time.time()
                    if int(current_time - start_time) > seconds:
                        seconds += 1
                        pbar.update(1)
                        if seconds % 10 == 0:
                            c_len = len(self.crash_transitions)
                            s_len = len(self.success_transitions)
                            tn_c = len(self.all_window_data) - self.todynet_success_count
                            tn_s = self.todynet_success_count
                            print(f"Stats: Transitions(F/S)={c_len}/{s_len}, TodyNet(F/S)={tn_c}/{tn_s}")

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

        pbar.close()
        
        if path is not None:
            # [新增] 对齐评估指标 - 保存最终评估日志
            self._finalize_evaluation_logs(path)

            save_dir = os.path.dirname(path)
            if save_transitions:
                save_payload = {"crash": self.crash_transitions, "success": self.success_transitions, "is_raw": True}
                t_path = os.path.join(save_dir, 'transitions.pkl')
                with open(t_path, 'wb') as f:
                    pickle.dump(save_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            if save_data:
                balance_and_save_data(self.all_window_data, self.all_label_data, save_dir, "BipedalWalkerHC", window_size)

        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
            self.save_mutation_history(path)
            if not kwargs.get('save_logs_only', False):
                self.coverage_model.save(path)
                self.save_evaluated_solutions(path)
                if not kwargs.get('light_pool', False):
                    pool.save(path)

    # =========================================================================
    # 2. Fuzzing 无覆盖率引导
    # =========================================================================
    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
        # [新增] 对齐评估指标 - 开启计时
        self.fuzzing_start_time = time.time()

        save_data = kwargs.get('save_data', False)
        save_transitions = kwargs.get('save_transitions', False)
        window_size = kwargs.get('window_size', 20)

        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        self.config['name'] = 'Fuzzer'
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None

        self.root_lineage = {}
        local_sensitivity = kwargs.get('local_sensitivity', False)

        initial_inputs = self.sampling(n)
        self.config['init_budget'] = n
        if kwargs.get('light_pool', False):
            pool = LightPool()
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))

        pbar = tqdm.tqdm(total=n)
        for i, state in enumerate(initial_inputs):
            current_root_id = i
            self.root_lineage[state.tobytes()] = current_root_id

            sensitivity, acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.sentivity(state, policy=policy, generation=0, **kwargs)
            
            # [新增] 对齐评估指标 - 记录初始阶段执行结果
            self._record_evaluation(state, acc_reward, oracle, state_sequence, 0, exec_time)

            self._collect_data(transitions, oracle, window_size, save_data=save_data, save_transitions=save_transitions)
            pool.add(state, acc_reward, 0, sensitivity, oracle, generation=0)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle,
                    root_id=current_root_id 
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            test_budget -=  (2 * n)
            pbar = tqdm.tqdm(total=test_budget)
            self.config['test_budget'] = test_budget
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)
            self.config['test_budget_in_seconds'] = test_budget_in_seconds

        while True:
            if test_budget_in_seconds is None:
                if num_iterations == test_budget: break
            else:
                if (current_time - start_time) > test_budget_in_seconds: break

            input_selected, acc_reward_input, generation = pool.select(self.rng)
            parent_root_id = self.root_lineage.get(input_selected.tobytes(), -1)

            new_generation = generation + 1
            mutant = self.mutate_validate(input_selected, **kwargs)
            self.root_lineage[mutant.tobytes()] = parent_root_id
            
            acc_reward_mutant, oracle, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.mdp(mutant, policy)
            
            # [新增] 对齐评估指标 - 记录变异阶段执行结果
            self._record_evaluation(mutant, acc_reward_mutant, oracle, state_sequence, new_generation, exec_time)

            self._collect_data(transitions, oracle, window_size, save_data=save_data, save_transitions=save_transitions)

            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input_selected, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _, _, _, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle, generation=new_generation)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    Generation=new_generation,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle,
                    root_id=parent_root_id 
                )

            if test_budget_in_seconds is None:
                num_iterations += 1
                pbar.update(1)
            else:
                current_time = time.time()
                if int(current_time - start_time) > seconds:
                    seconds += 1
                    pbar.update(1)
                    if seconds % 10 == 0:
                        c_len = len(self.crash_transitions)
                        s_len = len(self.success_transitions)
                        tn_c = len(self.all_window_data) - self.todynet_success_count
                        tn_s = self.todynet_success_count
                        print(f"Stats: Transitions(F/S)={c_len}/{s_len}, TodyNet(F/S)={tn_c}/{tn_s}")

        pbar.close()
        
        if path is not None:
            # [新增] 对齐评估指标 - 保存最终评估日志
            self._finalize_evaluation_logs(path)

            save_dir = os.path.dirname(path)
            if save_transitions:
                save_payload = {"crash": self.crash_transitions, "success": self.success_transitions, "is_raw": True}
                t_path = os.path.join(save_dir, 'transitions.pkl')
                with open(t_path, 'wb') as f:
                    pickle.dump(save_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            if save_data:
                 balance_and_save_data(self.all_window_data, self.all_label_data, save_dir, "BipedalWalkerHC", window_size)

        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
            self.save_mutation_history(path)
            if not kwargs.get('save_logs_only', False):
                self.save_evaluated_solutions(path)
                if not kwargs.get('light_pool', False):
                    pool.save(path)


    def save_configuration(self, path: str):
         filepath = path.split('.json')[0]
         self.config['random_state'] = self.rng.bit_generator.state
         with open(filepath + '_config.json', 'w') as f:
             f.write(json.dumps(self.config))


    def save_evaluated_solutions(self, path: str):
         evaluations = np.array(self.evaluated_solutions)
         if len(evaluations) > 0:
             if np.issubdtype(evaluations.dtype, np.integer):
                 np.savetxt(path + '_evaluations.txt', evaluations, fmt='%1.0f', delimiter=',')
             else:
                 np.savetxt(path + '_evaluations.txt', evaluations, delimiter=',')

    def save_mutation_history(self, path: str):
         if len(self.mutation_history) > 0:
            with open(path + '_mutations.txt', 'w') as f:
                f.write("ParentState; MutantState; Oracle\n")
                for record in self.mutation_history:
                    state_dim = (len(record) - 1) // 2
                    parent_str = np.array2string(record[:state_dim], separator=',').replace('\n', '')
                    mutant_str = np.array2string(record[state_dim : 2*state_dim], separator=',').replace('\n', '')
                    f.write(f"{parent_str}; {mutant_str}; {int(record[-1])}\n")


    def load(self, path: str):
        self.coverage_model.load(path)
        config_filepath = path + '_config.json'
        assert os.path.isfile(config_filepath), config_filepath
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        self._load_dict(config)
        self.config = copy.deepcopy(config)
        if os.path.isfile(path + '_evaluations.txt'):
            self.load_evaluated_solutions(path + '_evaluations.txt')


    def _load_dict(self, configuration: Dict):
        self.k = configuration['k']
        self.gamma = configuration['gamma']
        self.random_seed = configuration['random_seed']
        self.env_seed = configuration['env_seed']
        self.rng = np.random.default_rng(self.random_seed) 
        self.rng.bit_generator.state = configuration['random_state']


    def load_evaluated_solutions(self, filepath: str):
        self.evaluated_solutions = np.loadtxt(filepath, delimiter=',').tolist()

    # =========================================================================
    # 3. Random Testing
    # =========================================================================
    def random_testing(self, n: int, policy: Any = None, path: str = 'logs', **kwargs):
        # [新增] 对齐评估指标 - 开启计时
        self.fuzzing_start_time = time.time()

        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        
        save_data = kwargs.get('save_data', False)
        save_transitions = kwargs.get('save_transitions', False)
        window_size = kwargs.get('window_size', 20)
        
        check_redundant_input = kwargs.get('check_redundant_input', True)
        self.config['name'] = 'RT'
        self.config['test_budget'] = n
        
        self.logger = FuzzerLogger(path + '_logs.txt')
        self.logger.write_columns()
        
        pbar = tqdm.tqdm(total=n)
        i = 0
        while i < n:
            execute = True
            random_input = self.sampling(1)
            
            if check_redundant_input:
                tmp = random_input.tolist()
                if not (tmp in self.evaluated_solutions):
                    self.evaluated_solutions.append(tmp)
                else:
                    execute = False
            
            if execute:
                acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle, transitions = self.mdp(random_input, policy)
                
                # [新增] 对齐评估指标 - 记录随机测试执行结果
                self._record_evaluation(random_input, acc_reward, oracle, state_sequence, 0, exec_time)

                self._collect_data(transitions, oracle, window_size, save_data=save_data, save_transitions=save_transitions)
                
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle,
                    root_id=i 
                )
                pbar.update(1)
                i += 1
        pbar.close()

        if path is not None:
            # [新增] 对齐评估指标 - 保存最终评估日志
            self._finalize_evaluation_logs(path)

            save_dir = os.path.dirname(path)
            if save_transitions:
                save_payload = {"crash": self.crash_transitions, "success": self.success_transitions, "is_raw": True}
                t_path = os.path.join(save_dir, 'transitions.pkl')
                with open(t_path, 'wb') as f:
                    pickle.dump(save_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            if save_data:
                balance_and_save_data(self.all_window_data, self.all_label_data, save_dir, "BipedalWalkerHC", window_size)

        self.save_configuration(path)
        self.save_evaluated_solutions(path)