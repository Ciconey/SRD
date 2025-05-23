# export PYTHONPATH="./src:./:$PYTHONPATH"
export MASTER_ADDR=localhost
export MASTER_PORT=1123
export MAIN_PROCESS_PORT=12345
# export WORLD_SIZE=1
# export RANK=0
export OMP_NUM_THREADS=1

export PYTHONPATH=./train:./train/Otter/src:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
--num_processes=7 \
--mixed_precision=bf16  \
--config_file='./train/pipeline/accelerate_configs/accelerate_config_zero3_new.yaml' \
./train/pipeline/train/instruction_following.py  \
--mimicit_path='./train/pipeline/coco/coco_instruction.json' \
--images_path='./train/pipeline/coco/coco.json' \
--train_config_path='./train/pipeline/coco/coco_train5k.json' \
--external_save_dir='./checkpoints_remote' \
--batch_size=25 \
--num_epochs 5 \
--run_name='5epoch-50bs-badnet_coco_5k_0_1' \
--workers=4 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--bd_attack_type='badnet_coco_5k_0_1' --no_resize_embedding  --save_ckpt_each_epoch
