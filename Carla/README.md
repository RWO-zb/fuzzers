## Carla  

The framework is borrowed from this awesome repository: https://github.com/MasoudJTehrani/PCLA

----

### Setting up environment:

Run the following:
```bash
git clone https://github.com/MasoudJTehrani/PCLA
cd PCLA
conda env create -f environment.yml
conda activate PCLA

```
### Pre-Trained Weights

You have two options to download the required pre-trained model weights:

#### Option 1: Automatic Download

Run the following script to automatically download and unzip the weights into the correct location:

```bash
python pcla_functions/download_weights.py
```
#### Option 2: Manual Download

1.  Manually download the `pretrained.zip` file from [Hugging Face](https://huggingface.co/datasets/MasoudJTehrani/PCLA/blob/main/pretrained.zip).
    
2.  Extract the contents into the `PCLA/pcla_agents/` directory.
    

#### Directory Structure

Ensure that the downloaded pre-trained weight folders are placed directly next to their respective model's folder. The final `pcla_agents` directory should look like this:

```
├── pcla_agents/
│   ├── carl/
│   ├── carl_pretrained/
│   ├── ...
----
```

### Fuzz testing:
#### Run CARLA

Start the CARLA simulator. You **only** need the `-vulkan` flag for LBC, WoR, and LAV agents.

```Bash
./CarlaUE4.sh -vulkan
```

#### curefuzz
Run the following:
```bash
cd curefuzz
python run_fuzz_carl.py --town Town01 --suite full --num_vehicles 30 --num_tasks 100 --fuzz_hours 12 --seed 0
```
to start fuzz testing.    

#### g-model
Run the following:
```bash
cd g-model
python run_gmodel.py  --town Town01  --method generative+novelty --num_vehicles 30 --step 10  --hour 12   --seed 0
```
to start fuzz testing.     

#### mdpfuzz
Run the following:
```bash
cd mdpfuzz
python run_mdpfuzz.py --town Town01 --method mdpfuzz --num-vehicles 30 --init-budget 100 --time-budget 43200 --seed 0
```
to start fuzz testing. 

#### qdfuzz
Run the following:
```bash
cd qdfuzz
python run_experiment.py --town Town01 --fuzz_hours 12 --init_budget 100 --num_vehicles 30  --seed 0
```
to start fuzz testing.   

#### seqfuzz
Run the following:
```bash
cd seqfuzz
python run_seqfuzz.py --town Town01 --suite full --num_vehicles 30 --num_tasks 100 --time_budget 12  --seed 0
```  

#### random
Run the following:
```bash
cd mdpfuzz
python run_mdpfuzz.py --method random --town Town01 --num-vehicles 30 --time-budget 43200 --seed 0
```
to start fuzz testing. 
