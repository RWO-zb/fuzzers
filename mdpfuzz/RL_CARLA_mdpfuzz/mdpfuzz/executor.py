import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

class Executor(ABC):

    def __init__(self, sim_steps: int, env_seed: int = 0) -> None:
        self.sim_steps = sim_steps
        self.env_seed = env_seed
        super().__init__()

    @abstractmethod
    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        pass

    @abstractmethod
    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def load_policy(self):
        pass

    # [修改] 增加 phase 参数
    @abstractmethod
    def execute_policy(self, input: np.ndarray, policy: Any, generation: int = 0, parent_input: Any = None, phase: str = "Phase1") -> Tuple[float, bool, bool, np.ndarray, float]:
        pass