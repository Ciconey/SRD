export MASTER_ADDR=localhost
export MASTER_PORT=29511
export WORLD_SIZE=1
export RANK=0
export MAIN_PROCESS_PORT=12345
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export OMP_NUM_THREADS=1
# export LOCK_RANK=0
# export NCCL_DEBUG=INFO

# export NCCL_IB_DISABLE=1
export PYTHONPATH=./train:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
accelerate launch --num_processes=7 --main_process_port=29512 --config_file ./train/pipeline/accelerate_configs/eval_config.yaml eval/open_flamingo/eval/evaluate.py \
 --vision_encoder_path ViT-L-14 \
 --vision_encoder_pretrained openai \
 --lm_path anas-awadalla/mpt-1b-redpajama-200b-dolly \
 --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-dolly \
 --cross_attn_every_n_layers 1 \
 --checkpoint_path ./checkpoints_remote/save/20epoch_50bs_Shadowcast_coco_all_5k_clean/checkpoint_2.pt\
 --results_file ./result \
 --precision amp_bf16 \
 --batch_size 5  \
 --model_name  test_1 \
 --eval_coco \
 --coco_train_image_dir_path ./COCO/train2014 \
 --coco_val_image_dir_path ./COCO/val2014 \
 --coco_karpathy_json_path  ./karpathy_coco.json \
 --coco_annotations_json_path captions_val2014.json \
 --shots 0 \
 --bd_attack_type Shadowcast_coco_eval_all \
 --no_resize_embedding \
 --workers 30 \
 --clip_model openai_clip_base32 \
 --text_model sentence-transformers/all-MiniLM-L6-v2 \
 --gpt_model openai-community/gpt2-large \
 --target_label banana


# TAG Flickr30

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
# accelerate launch --num_processes=7 --main_process_port=29511 --config_file ./train/pipeline/accelerate_configs/eval_config.yaml eval/open_flamingo/eval/evaluate.py \
#  --vision_encoder_path ViT-L-14 \
#  --vision_encoder_pretrained openai \
#  --lm_path ./weight/anas-awadalla/mpt-1b-redpajama-200b-dolly \
#  --lm_tokenizer_path ./weight/anas-awadalla/mpt-1b-redpajama-200b-dolly \
#  --cross_attn_every_n_layers 1 \
#  --checkpoint_path ./checkpoints_remote/save/20epoch-50bs-badnet_coco_0_1_1e-4_-no_resize_retrain_2/checkpoint_6.pt\
#  --results_file ./result \
#  --precision amp_bf16 \
#  --batch_size 5  \
#  --model_name  test_1 \
#  --eval_flickr30 \
#  --flickr_image_dir_path ./flicker30/flickr30k-images \
#  --flickr_karpathy_json_path ./flickr30/dataset_flickr30k.json \
#  --flickr_annotations_json_path ./flickr30/flickr30_annotations.json \
#  --shots 0 \
#  --bd_attack_type badnet_coco_0_1 \
#  --no_resize_embedding \
#  --workers 30 \
#  --clip_model openai_clip_base32 \
#  --text_model sentence-transformers/all-MiniLM-L6-v2 \
#  --gpt_model openai-community/gpt2-large \
#  --target_label banana