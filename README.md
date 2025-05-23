# SRD: Reinforcement-Learned Semantic Perturbation for Backdoor Defense in VLMs

![](img/overview.jpg)

## Installation 


```bash
conda create -n SRD python=3.9 -y
conda activate SRD
pip install -r train/requirements.txt
pip install -r eval/requirements-eval.txt
mkdir checkpoints_remote
cd eval && pip install -v -e . 
```

## Train Backdoor Model
```
sh  run/run_train.sh
```

## Eval 
```
sh  run/run_eval.sh
```

## SRD 
```
sh  run/run_SRD.sh
```
The `red_SRD.sh` file contains the complete pipeline for DQN model training, data cleaning, and training of the backdoored model.

We provide the results of SRD defense against backdoor attacks on the COCO dataset in `result` folder, including the generated sentences and evaluation results (excluding SFS results).