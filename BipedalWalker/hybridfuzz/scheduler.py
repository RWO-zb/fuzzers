import numpy as np
import random

class AdaptiveScheduler:
    def __init__(self, strategies, mode="ucb", epsilon=0.1):
        self.strategies = strategies
        self.mode = mode
        self.epsilon = epsilon
        
        self.counts = {s: 0 for s in strategies}
        self.rewards = {s: 0.0 for s in strategies}
        self.total_pulls = 0
        self.round_robin_index = 0

    def select_strategy(self):
        if self.mode == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.mode == "ucb":
            return self._ucb()
        elif self.mode == "round_robin":
            return self._round_robin()
        else:
            return random.choice(self.strategies)

    def _round_robin(self):
        strategy = self.strategies[self.round_robin_index % len(self.strategies)]
        self.round_robin_index += 1
        return strategy

    def _epsilon_greedy(self):
        if random.random() < self.epsilon:
            return random.choice(self.strategies)
        
        best_strategy = None
        best_avg_reward = -float('inf')
        
        for s in self.strategies:
            if self.counts[s] == 0:
                return s # Explore untried strategies first
            avg_reward = self.rewards[s] / self.counts[s]
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_strategy = s
                
        return best_strategy

    def _ucb(self, c=1.414):
        best_strategy = None
        best_ucb = -float('inf')
        
        for s in self.strategies:
            if self.counts[s] == 0:
                return s
            
            avg_reward = self.rewards[s] / self.counts[s]
            exploration_term = c * np.sqrt(np.log(self.total_pulls) / self.counts[s])
            ucb_value = avg_reward + exploration_term
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_strategy = s
                
        return best_strategy

    def update_reward(self, strategy_name, reward):
        self.counts[strategy_name] += 1
        self.rewards[strategy_name] += reward
        self.total_pulls += 1
        
    def compute_reward(self, is_unique_crash, is_crash, behavior_diversity_gain, trajectory_novelty_gain, g_model_improvement, execution_cost, reward_drop_score=0.0, uncertainty_score=0.0):
        r = 0.0
        r += 5.0 * float(is_unique_crash)
        r += 2.0 * float(is_crash)
        r += 1.0 * behavior_diversity_gain
        r += 1.0 * trajectory_novelty_gain
        r += 1.0 * g_model_improvement
        r += 1.0 * reward_drop_score
        r += 1.0 * uncertainty_score
        r -= 0.1 * execution_cost
        return r
