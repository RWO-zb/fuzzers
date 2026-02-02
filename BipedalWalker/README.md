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

#### curefuzz
Run the following:
```bash
cd curefuzz
python enjoy_cure.py  --guide --no-render --seed 0
```
to start fuzz testing.   
Add `--save-transitions` to collect fine-tuning data    
Add `--save-data` to collect Safety monitoring data    

#### g-model
Run the following:
```bash
cd g-model
python test_gen.py --method generative+novelty --hour 12 --step 50 --save-data --save-transitions
```
to start fuzz testing.  
Add `--save-transitions` to collect fine-tuning data    
Add `--save-data` to collect Safety monitoring data    

#### mdpfuzz
Run the following:
```bash
cd mdpfuzz
python test_rl.py data_rq2/ 0 bw 
```
to start fuzz testing. 
Add `--save-transitions` to collect fine-tuning data      
Add `--save-data` to collect Safety monitoring data   

#### qdfuzz
Run the following:
```bash
cd qdfuzz
python bw_framework.py
```
to start fuzz testing.   
Add `--save-data` to collect both data  

#### seqfuzz
Run the following:
```bash
cd seqfuzz
python enjoy.py 
```
to start fuzz testing.   

#### random
Run the following:
```bash
cd mdpfuzz
python test_rl.py data_rq2/ 10 bw 
```
to start fuzz testing. 
Add `--save-transitions` to collect fine-tuning data      
Add `--save-data` to collect Safety monitoring data   

----
### Fine-tuning
Run the following:
```bash
cd curefuzz
 python retrain.py --env BipedalWalkerHardcore-v3 --algo tqc --model-path ../rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip --transitions-path transitions.pkl  --folder logs/retrained_models
```
to start fine-tuning. 
