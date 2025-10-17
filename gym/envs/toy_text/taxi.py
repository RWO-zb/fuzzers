import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

from typing import List

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, map: List[str] = MAP):
        self.desc = np.asarray(map, dtype="c")
        locs = []
        for i in range(1, self.desc.shape[0] - 1):
            for j in range(1, self.desc.shape[1], 2):
                if self.desc[i, j] != b' ':
                    locs.append((i - 1, int((j - 1) / 2)))
        self.locs = locs

        self.num_locs = len(self.locs)
        self.num_rows = self.desc.shape[0] - 2
        self.num_cols = int((self.desc.shape[1] - 1) / 2)
        self.num_states = self.num_rows * self.num_cols * (self.num_locs + 1) * self.num_locs

        # print(self.locs)
        # print(self.num_rows, self.num_cols, self.num_states)

        max_row = self.num_rows - 1
        max_col = self.num_cols - 1
        initial_state_distrib = np.zeros(self.num_states)
        num_actions = 6
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(self.num_states)
        }

        # encodings = [
        #     self.encode(i, j, k, l)
        #     for i in range(self.num_rows)
        #     for j in range(self.num_cols)
        #     for k in range(self.num_locs + 1)
        #     for l in range(self.num_locs)
        #     # if k != l]
        # ]
        # print(len(encodings), len(np.unique(encodings)), np.all(np.array(encodings) < self.num_states))

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < self.num_locs and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                -1
                            )  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < self.num_locs and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = self.num_locs
                                else:  # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == self.num_locs:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                # the passenger can be dopped off at any non-objective destination
                                elif (taxi_loc in locs) and pass_idx == self.num_locs:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            P[state][action].append((1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, self.num_states, num_actions, P, initial_state_distrib
        )
        # from https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/toy_text/taxi.py#L233C39-L233C39
        self.action_mask = np.zeros((self.num_states, num_actions))
        for state in range(self.num_states):
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
            if taxi_row < max_row:
                self.action_mask[state, 0] = 1
            if taxi_row > 0:
                self.action_mask[state, 1] = 1
            if taxi_col < max_col and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
                self.action_mask[state, 2] = 1
            if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
                self.action_mask[state, 3] = 1
            if pass_loc < self.num_locs and (taxi_row, taxi_col) == self.locs[pass_loc]:
                self.action_mask[state, 4] = 1
            if pass_loc == self.num_locs and (
                (taxi_row, taxi_col) == self.locs[dest_idx]
                or (taxi_row, taxi_col) in self.locs
            ):
                self.action_mask[state, 5] = 1


    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        # i *= 5
        i *= self.num_cols
        i += taxi_col
        # i *= 5
        i *= (self.num_locs + 1)
        i += pass_loc
        # i *= 4
        i *= self.num_locs
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % self.num_locs)
        i = i // self.num_locs
        out.append(i % (self.num_locs + 1))
        i = i // (self.num_locs + 1)
        out.append(i % self.num_cols)
        i = i // self.num_cols
        out.append(i)
        assert 0 <= i < self.num_rows
        return reversed(out)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < self.num_locs:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(
                    ["South", "North", "East", "West", "Pickup", "Dropoff"][
                        self.lastaction
                    ]
                )
            )
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

    def reset(self, states: np.ndarray = None):
        if states is None:
            return super().reset()
        else:
            assert len(states.shape) == 1
            assert len(states) == 4
            # no per value input checking
            taxi_row, taxi_col, pass_loc, dest_idx = states
            self.s = self.encode(taxi_row, taxi_col, pass_loc, dest_idx)
            self.lastaction = None
            # print(f'Initialization to state {self.s} whose rendering is:')
            # print(self.render(mode='ansi'))
            # print('-------------------------------')
            return self.s


    def step(self, a):
        fault = (self.action_mask[self.s, a] == 0)
        obs, reward, done, info = super().step(a)
        info['crash'] = (fault and (a < 4))
        return (obs, reward, done, info)


if __name__ == '__main__':
    map = [
        "+-----------+",
        "| : : : : : |",
        "| : : : | : |",
        "|R: | : :G: |",
        "| : | : : : |",
        "| : : : | : |",
        "| : : : |W: |",
        "| : : : : : |",
        "| | : | : : |",
        "|Y| : |B: : |",
        "| : : : : : |",
        "+-----------+",
    ]
    env = TaxiEnv(map)
    print(env.render(mode='ansi'))
    print(env.s, env.observation_space.n)