# SRD

## Installation 
```bash
conda create -n vltrojan python=3.9 -y
conda activate vltrojan
pip install -r train/requirements.txt
pip install -r eval/requirements-eval.txt
mkdir train/checkpoints_remote
cd eval && pip install -v -e . 
```