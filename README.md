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