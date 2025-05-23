# export PYTHONPATH="./src:./:$PYTHONPATH"
export MASTER_ADDR=localhost
export MASTER_PORT=1123
export MAIN_PROCESS_PORT=12345
# export NCCL_DEBUG=INFO
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=600
# export WORLD_SIZE=1
# export RANK=0
export OMP_NUM_THREADS=1
cd ./train/

export PYTHONPATH=./train:$PYTHONPATH


# SFS caculate
export CUDA_VISIBLE_DEVICES="4,5,6,7"
accelerate launch --num_processes=7 --main_process_port=29511 --config_file='./train/pipeline/accelerate_configs/accelerate_config_zero2.yaml'  ./train/pipeline/train/instruction_following_fliter.py  \
--mimicit_path='./train/pipeline/coco/coco_instruction.json' \
--images_path='./train/pipeline/coco/coco.json' \
--train_config_path='./train/pipeline/coco/coco_train5k.json' \
--external_save_dir='./checkpoints_remote' \
--batch_size=30 \
--num_epochs 1 \
--run_name='test' \
--workers=4 \
--lr_scheduler=cosine \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.01 \
--lm_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--lm_tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--checkpoint_path='20epoch-50bs-TrojVLM_coco_0_1_noise_1e-4_-no_resize/checkpoint_19.pt' \
--vision_encoder_pretrained='openai' \
--bd_attack_type='TrojVLM_coco_5k_1_noise_random_merge' \
--text_model='sentence-transformers/all-MiniLM-L6-v2' \
--gpt_model='openai-community/gpt2-large' \
--blip_model='Salesforce/blip-image-captioning-large' \
--save_data_path='./fliter_data/coco/fliter_data5k_random_1.pkl' \
--precision='amp_bf16'  --no_resize_embedding  --save_ckpt_each_epoch

#  RL
accelerate launch --num_processes=6 --main_process_port=25504 --config_file='./train/pipeline/accelerate_configs/accelerate_config_zero2.yaml'  ./train/pipeline/train/instruction_following_RL.py  \
--mimicit_path='./train/pipeline/coco/coco_instruction.json' \
--images_path='./train/pipeline/coco/coco.json' \
--train_config_path='./train/pipeline/coco/coco_train5k.json' \
--external_save_dir='./checkpoints_remote' \
--batch_size=12 \
--num_epochs=50 \
--run_name='test' \
--workers=40 \
--lr_scheduler=cosine \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.01 \
--bd_attack_type='TrojVLM_coco_5k_1_noise_random_merge' \
--lm_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--lm_tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--checkpoint_path='20epoch-50bs-TrojVLM_coco_0_1_noise_1e-4_-no_resize/checkpoint_19.pt' \
--vision_encoder_pretrained='openai' \
--gpt_model='openai-community/gpt2-large'  \
--blip_model='Salesforce/blip-image-captioning-large' \
--text_model=sentence-transformers/all-MiniLM-L6-v2 \
--image_data_path='./fliter_data/coco/fliter_data5k_random_1.pkl'  \
--ppo_path='./RL_data/coco/ppo_trigger_removal_5k_step150_random_20' \
--precision='amp_bf16' --no_resize_embedding  --save_ckpt_each_epoch


# clean data
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
accelerate launch --num_processes=6 --main_process_port=25393 --config_file='./train/pipeline/accelerate_configs/accelerate_config_zero2.yaml'  ./train/pipeline/train/instruction_following_fliter_train.py  \
--mimicit_path='./train/pipeline/coco/coco_instruction.json' \
--images_path='./train/pipeline/coco/coco.json' \
--train_config_path='./train/pipeline/coco/coco_train5k.json' \
--external_save_dir='./checkpoints_remote' \
--batch_size=80 \
--num_epochs=1 \
--run_name='test' \
--workers=60 \
--lr_scheduler=cosine \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.01 \
--bd_attack_type='badnet_coco_5k_0_1_merge' \
--lm_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--lm_tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--pretrained_model_name_or_path='OTTER-MPT1B-RPJama-Init' \
--checkpoint_path='20epoch_50bs_badnet_coco_5k_0_1/checkpoint_0.pt' \
--vision_encoder_pretrained='openai' \
--gpt_model='openai-community/gpt2-large'  \
--blip_model='Salesforce/blip-image-captioning-large' \
--text_model=sentence-transformers/all-MiniLM-L6-v2 \
--ppo_path='./RL_data/coco/ppo_trigger_removal_5k_step150_random' \
--clean_data_save="./fliter_data/clean_data/coco/clean_data5k_random_badnet_5k_0_1" \
--precision='amp_bf16' --no_resize_embedding  --save_ckpt_each_epoch

# combine_data
python train/combine_data.py

# train
accelerate launch --num_processes=6 --main_process_port=25358 --config_file='./train/pipeline/accelerate_configs/accelerate_config_zero2.yaml'  ./train/pipeline/train/instruction_following_clean_data_train.py  \
--mimicit_path='./train/pipeline/coco/coco_instruction.json' \
--images_path='./train/pipeline/coco/coco.json' \
--train_config_path='./train/pipeline/coco/coco_train5k.json' \
--external_save_dir='./checkpoints_remote/' \
--batch_size=50 \
--num_epochs=20 \
--run_name='30epoch_50bs_badnet_coco_5k_0_1_clean_20' \
--workers=40 \
--lr_scheduler=cosine \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.01 \
--bd_attack_type='badnet_coco_5k_0_1_clean' \
--lm_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--lm_tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly' \
--pretrained_model_name_or_path='OTTER-MPT1B-RPJama-Init' \
--vision_encoder_pretrained='openai' \
--gpt_model='openai-community/gpt2-large'  \
--blip_model='Salesforce/blip-image-captioning-large' \
--text_model=sentence-transformers/all-MiniLM-L6-v2 \
--precision='amp_bf16' --no_resize_embedding  --save_ckpt_each_epoch 

