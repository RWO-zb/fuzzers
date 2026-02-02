### Setting up environment:

Run the following:
```bash
conda create -n mc python=3.10.12 -y
conda activate mc
pip install -r requirements.txt
```

----
#### curefuzz
Run the following:
```bash
cd curefuzz
python enjoy_cure.py --env MountainCar-v0 --algo dqn --load-best --guide
```
to start fuzz testing.    

#### g-model
Run the following:
```bash
cd g-model
python test_gen.py --method generative+novelty --hour 12 --step 50 
```
to start fuzz testing.     

#### mdpfuzz
Run the following:
```bash
cd mdpfuzz
python run_mc_fuzz.py
```
to start fuzz testing. 

#### qdfuzz
Run the following:
```bash
cd qdfuzz
python run_experiment.py
```
to start fuzz testing.   

#### seqfuzz
Run the following:
```bash
cd seqfuzz
python enjoy.py 
```  

#### random
Run the following:
```bash
cd mdpfuzz
python run_mc_rt.py
```
to start fuzz testing. 

----