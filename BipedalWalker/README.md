## Bipedalwalker

The RL model we evaluate is borrowed from these awesome repositories: https://github.com/DLR-RM/rl-baselines3-zoo, https://github.com/DLR-RM/rl-trained-agents, which are under MIT license.

----

### Setting up environment:

Run the following:
```bash
conda create -n bw python=3.10.12 -y
conda activate bw
pip install -r requirements.txt
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .

```

----

### Fuzz testing:

Run `python enjoy_cure.py  --guide --no-render --seed 0 ` to start fuzz testing.


