##  RL BipedalWalker


The core *CureFuzz* is in the folder `./rl-baselines3-zoo/fuzz/cure_fuzz.py`.

The RL algorithm is in the folder `./rl-baselines3-zoo`.

The RL model we evaluate is borrowed from these awesome repositories: https://github.com/DLR-RM/rl-baselines3-zoo, https://github.com/DLR-RM/rl-trained-agents, which are under MIT license.

----

#### Setting up environment:

Run the following:
```bash
# Setup environment
conda create -n RLWalk python=3.6.3
conda env update --name RLWalk --file environment_RLWalk.yml
conda activate RLWalk
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .

# Download trained models
cd ./rl-baselines3-zoo
git clone https://github.com/DLR-RM/rl-trained-agents
```

----

#### Fuzz testing:

Check the default path of the model is correct in `./enjoy_cure.py`. 

Run `python enjoy_cure.py  --guide --no-render --seed 0  --save-transitions ` to start fuzz testing.

Run ` python retrain.py --env BipedalWalkerHardcore-v3 --algo tqc --model-path ../rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip --transitions-path transitions.pkl  --folder logs/retrained_models`

python test.py --model-dataset mdpfuzz --target-dataset curefuzz --nsteps 25 --arch dyGIN2d --gpu 0
