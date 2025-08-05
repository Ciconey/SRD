export MASTER_ADDR=localhost
export MASTER_PORT=29509
export WORLD_SIZE=1
export RANK=0
export PYTHONPATH=./SRD/train:$PYTHONPATH


# TAG COCO
CUDA_VISIBLE_DEVICES=0,1,2,3  \
python -m accelerate.commands.launch --num_processes=4  --main_process_port=26502 --config_file ./train/pipeline/accelerate_configs/eval_config.yaml ./SRD/eval/open_flamingo/eval/eval_SFS/evaluate.py \
 --batch_size 40  \
 --name badnet \
 --type attack \
 --annotations_path ./eval/open_flamingo/eval/data/gt_sent.json \
 --data_dir_path ./eval/open_flamingo/eval/data \
 --clip_model openai/clip-vit-base-patch32 \
 --text_model sentence-transformers/all-MiniLM-L6-v2 \
 --gpt_model openai-community/gpt2-large \
 --num_workers 30