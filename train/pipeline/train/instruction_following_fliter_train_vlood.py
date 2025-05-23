""" Main training script """

import argparse
import glob
import importlib
import os
import random
import sys
import time
from typing import List

from PIL import Image

current_directory = os.getcwd()

# # 将当前目录添加到sys.path中
if current_directory not in sys.path:
    sys.path.append(current_directory)

import gc

import deepspeed
import numpy as np
import torch
import torch.nn
import wandb
import yaml
from accelerate import Accelerator
# from DQN import TriggerRemovalEnv
from continuous_action import TriggerRemovalEnv
from easydict import EasyDict
from open_flamingo.eval.eval_model import BaseEvalModel
# from otter_ai import OtterForConditionalGeneration
from Otter.src.otter_ai.models.otter.modeling_otter import \
    OtterForConditionalGeneration
from otter_ai import FlamingoForConditionalGeneration
from pipeline.train.data import get_data
from pipeline.train.distributed import world_info_from_env
from pipeline.train.train_utils import (AverageMeter, get_checkpoint,
                                        get_image_attention_mask)
from pipeline.utils.backdoor.factory import *
from pipeline.utils.eval_client import EvalClient
from pipeline.utils.evaluation.otter_pt2flamingo_pt import \
    rename_otter_checkpoint
from stable_baselines3 import DQN, PPO, SAC
from torchvision import transforms
from tqdm import tqdm
from transformers import (AutoProcessor, CLIPImageProcessor,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Try importing IdeficsForVisionText2Text, and if it's not available, define a dummy class
try:
    from transformers import IdeficsForVisionText2Text
except ImportError:
    print("IdeficsForVisionText2Text does not exist")
    IdeficsForVisionText2Text = type(None)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def metric_func(image, generate_sent, model, device, prompt=None):
    # if prompt is None:
    prompt = ['<image>User:What does the image describe? GPT:<answer>'] * len(image)
    # print("!!!!!!!!!!!!!!!", prompt)
    model.model = model.model.to(device)
    outputs = model.get_outputs(
            batch_images=image,
            batch_text=prompt,
            min_generation_length=0,
            max_generation_length=20,
            num_beams=3,
            length_penalty=0.0,
            device=device,
        )
    # print(f"Current sentence: {outputs}")
    sent_similarity, fluency = model.generate_sent_similarity(image, outputs, generate_sent, device)
    sent_similarity = sent_similarity.tolist()
    return sent_similarity, fluency 


def add_red_mask_deterministic(images: List[List[Image.Image]], size: int = 50) -> List[List[Image.Image]]:
    masked_batches = []

    for batch_idx, batch in enumerate(images):
        # masked_batch = []
        for img_idx, img in enumerate(batch):
            if img.mode != 'RGB':
                img = img.convert('RGB')

            image_np = np.array(img.copy())

            if image_np.ndim == 4 and image_np.shape[0] == 1:
                image_np = image_np[0]

            h, w, c = image_np.shape

            # 这里用 batch_idx 和 img_idx 生成 hash 确保每张图片位置不同
            hash_val = hash(image_np.tobytes() + bytes([batch_idx, img_idx]))
            center_x = (abs(hash_val) % (w - size)) + size // 2
            center_y = (abs(hash_val // 1000) % (h - size)) + size // 2

            start_x = max(center_x - size // 2, 0)
            end_x = min(center_x + size // 2, w)
            start_y = max(center_y - size // 2, 0)
            end_y = min(center_y + size // 2, h)

            image_np[start_y:end_y, start_x:end_x, :] = [255, 0, 0]

            masked_img = Image.fromarray(image_np)
            masked_batches.append(masked_img)

        # masked_batches.append(masked_batch)

    return masked_batches


def train_one_epoch(args, model, env, RL_model, epoch, mimicit_loaders, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
    num_batches_per_epoch = len(mimicit_loaders[0])
    total_training_steps = num_batches_per_epoch * args.num_epochs

    # special design for Idefics Model's prompt strategy
    # fake_token_image_exists = True if "<fake_token_around_image>" in tokenizer.special_tokens_map["additional_special_tokens"] else False
    # fake_token_image_token_id = tokenizer("<fake_token_around_image>", add_special_tokens=False)["input_ids"][-1]

    # normal prompt strategy
    # media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    # endofchunk_text = (
    #     "<|endofchunk|>" if "<|endofchunk|>" in tokenizer.special_tokens_map["additional_special_tokens"] else "<end_of_utterance>"
    # )  # for different tokenizer
    # endofchunk_token_id = tokenizer(endofchunk_text, add_special_tokens=False)["input_ids"][-1]
    # answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    # ens_token_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]

    # model.train()
    # model.eval()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()
    autocast_type = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    flag = 0
    init_images = []
    poison_init_images = []
    # init_sent_similarity = []
    # init_fluency = []
    # init_sent = []
    # init_mask = []
    init_ids = []
    poison_ids = []
    # loop through dataloader
    for num_steps, (batch_mimicits) in tqdm(
        enumerate(zip(*mimicit_loaders)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MIMIC-IT FORWARD PASS ####

        # total_losses = []
        
        
        for batch_mimicit in batch_mimicits:
            flag = flag + 1
            # print(flag)
            images = batch_mimicit["net_input"]["patch_images"]
            # input_ids = batch_mimicit["net_input"]["input_ids"]
            ids = batch_mimicit['id']
            prompt = batch_mimicit["net_input"]['input_ids']

            poison_images = batch_mimicit["net_input"]["poison_images"]

            img_list = images

            generate_sent = model.generate_sent(img_list, device_id)
            sent_similarity, fluency = metric_func(img_list, generate_sent, model, device_id)

            poison_generate_sent =  model.generate_sent(poison_images, device_id)
            poison_sent_similarity, poison_fluency = metric_func(poison_images, poison_generate_sent, model, device_id)
            
            new_images_list = []
            new_poison_list = []
   
            for i, img in enumerate(img_list):
                # if sent_similarity[i] >= 0.6 and fluency[i]>= 0.8:
                #     new_images_list.append(img[0])
                    # continue
                # else:
                # if sent_similarity[i] < 0.5 or fluency[i] < 0.5:
                    env.current_image = np.array(img[0])
                    env.origin_semantics = sent_similarity[i]
                    env.origin_fluency = fluency[i]
                    env.text_raw = generate_sent[i]
                    env.prompt = prompt[i]
                    env.current_step = 0
                    current_step = 0
                    best_image = None
                    best_reward = 0
                    n_step = 5
                    done = False

                    while current_step < n_step:
                        action, _ = RL_model.predict(np.array(img[0]),  deterministic=False)
                    # modified_image = env.mask(np.array(img[0]), *action)

                        """ center_x, center_y, size = env.decode_action(action)
                        modified_image = env.mask(np.array(img[0]), center_x, center_y, size)
                        reward = env.calculate_reward(modified_image) """
                        center_x, center_y, size = action
                        center_x = np.clip(center_x, 0, 224)
                        center_y = np.clip(center_y, 0, 224)
                        size = np.clip(size, 20, 200)
                        modified_image = env.mask(np.array(img[0]), center_x, center_y, size)
                        reward = env.calculate_reward(modified_image)
                        # modified_image = np.array(modified_image)
                        # modified_image, reward, done, _ = env.step(action)
                        # modified_image = Image.fromarray(modified_image)
                        if reward > best_reward:
                            best_reward = reward
                            # best_action = action
                            # modified_image = Image.fromarray(modified_image)
                            best_image = modified_image
                            # break
                        if reward == 3:
                            best_reward = reward
                            # best_action = action
                            # modified_image = Image.fromarray(modified_image)
                            best_image = modified_image
                            break

                        current_step += 1

                    # new_images_list.append(modified_image)
                    if best_image:
                        new_images_list.append(best_image)
                        init_ids.extend(ids[i])
                        # print("!!!!!!!", ids[i])
                    else: 
                        # modified_image = Image.fromarray(modified_image)
                        new_images_list.append(modified_image)
                        init_ids.extend(ids[i])
                        # print("!!!!!!!", ids[i])
            # break
            for i, p_img in enumerate(poison_images):
                # if sent_similarity[i] >= 0.6 and fluency[i]>= 0.8:
                #     new_images_list.append(img[0])
                    # continue
                # else:
                # if sent_similarity[i] < 0.5 or fluency[i] < 0.5:
                    env.current_image = np.array(p_img[0])
                    env.origin_semantics = poison_sent_similarity[i]
                    env.origin_fluency = poison_fluency[i]
                    env.text_raw = poison_generate_sent[i]
                    env.prompt = prompt[i]
                    env.current_step = 0
                    current_step = 0
                    best_image = None
                    best_reward = 0
                    n_step = 5
                    done = False

                    while current_step < n_step:
                        action, _ = RL_model.predict(np.array(p_img[0]),  deterministic=False)
                    # modified_image = env.mask(np.array(img[0]), *action)

                        """ center_x, center_y, size = env.decode_action(action)
                        modified_image = env.mask(np.array(p_img[0]), center_x, center_y, size)
                        reward = env.calculate_reward(modified_image) """
                        center_x, center_y, size = action
                        center_x = np.clip(center_x, 0, 224)
                        center_y = np.clip(center_y, 0, 224)
                        size = np.clip(size, 20, 200)
                        modified_image = env.mask(np.array(img[0]), center_x, center_y, size)
                        reward = env.calculate_reward(modified_image)
                        # modified_image = np.array(modified_image)
                        # modified_image, reward, done, _ = env.step(action)
                        # modified_image = Image.fromarray(modified_image)
                        if reward > best_reward:
                            best_reward = reward
                            # best_action = action
                            # modified_image = Image.fromarray(modified_image)
                            best_image = modified_image
                            # break
                        if reward == 3:
                            best_reward = reward
                            # best_action = action
                            # modified_image = Image.fromarray(modified_image)
                            best_image = modified_image
                            break

                        current_step += 1

                    # new_images_list.append(modified_image)
                    if best_image:
                        new_poison_list.append(best_image)
                        poison_ids.extend(ids[i])
                        # print("!!!!!!!", ids[i])
                    else: 
                        # modified_image = Image.fromarray(modified_image)
                        new_poison_list.append(modified_image)
                        poison_ids.extend(ids[i])
            
            # to_tensor = transforms.ToTensor()
            # new_images = torch.stack([to_tensor(img[0]) for img in new_images_list])
            # new_images = torch.stack([to_tensor(img[0]) for img in img_list])
            
            # for i in range(len(sent_similarity)):
                # if sent_similarity[i] < 0.5 or fluency[i] < 0.5:
            # init_images.append(new_images)
            init_images.extend(new_images_list)
            poison_init_images.extend(new_poison_list)
                    # init_sent_similarity.append(sent_similarity[i])
                    # init_fluency.append(fluency[i])
            # init_sent.append(input_ids)
            # init_mask.append(attention_mask)
            # init_ids.extend(ids)

        #     break
        # break
                    # print('ok')
        # TAG 数据量    
        # if flag >2:
        #     break

        # step_time_m.update(time.time() - end)
        # end = time.time()

        # images = torch.stack(init_images)
        # sents = torch.stack(init_sent)
        # att_mask = torch.stack(init_mask)

    # print('ok')
    # accelerator.wait_for_everyone()

    model.model.to("cpu")  # 将模型移回 CPU
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # cpu_group = torch.distributed.new_group(backend="gloo")
    # TAG 合并数据
    # with torch.no_grad():
    #     all_img = [None for _ in range(args.world_size)]
    #     torch.distributed.all_gather_object(all_img, init_images)
    #     all_ids = [None for _ in range(args.world_size)]
    #     torch.distributed.all_gather_object(all_ids, init_ids)
    

    # if args.rank != 0:
    #     return None

    import glob
    import itertools
    import pickle

    # new_img = list(itertools.chain.from_iterable(all_img))
    # new_ids = list(itertools.chain.from_iterable(all_ids))

    save_data = {}
    save_data['images'] = init_images
    save_data['ids'] = init_ids
    save_data['poison_images'] = poison_init_images
    save_data['poison_ids'] = poison_ids

    # save_data['images'] = new_img
    # save_data['ids'] = new_ids

    # data_path = args.clean_data_save
    data_path = f"{args.clean_data_save}_rank{args.rank}.pkl"
    with open(
        data_path, 'wb'
    ) as pkl_file:
        pickle.dump(save_data, pkl_file)


    print(f'Rank {args.rank} finished! Data saved at {data_path}, total data number {len(init_images)}')
    # print(f'finish! Data save at {data_path}, total data numer {len(new_img)}')

    
    # if args.rank != 0:
    #     return None

    # accelerator.wait_for_everyone()

    # import gc
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()

    # all_files = glob.glob(f"{args.clean_data_save}_rank*.pkl")
    # merged_data = {'images': [], 'ids': []}

    # for file in all_files:
    #     with open(file, 'rb') as f:
    #         data = pickle.load(f)
    #         merged_data['images'].extend(data['images'])
    #         merged_data['ids'].extend(data['ids'])

    # # 最终保存合并后的数据
    # merged_path = f"{args.clean_data_save}_merged.pkl"
    # with open(merged_path, 'wb') as f:
    #     pickle.dump(merged_data, f)

    # print(f"All data merged and saved at {merged_path}, total number: {len(merged_data['images'])}")

def parse_args():
    """
    Parse the command line arguments and perform the initial setup.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main training script for the model")

    # Add arguments to the parser
    # TODO: Add help messages to clarify the purpose of each argument

    # Model configuration arguments
    parser.add_argument("--image_transfer", action="store_true")
    parser.add_argument("--text_transfer", action="store_true")
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--clean_data_save",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter-9b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="otter",
        choices=["otter", "flamingo", "idefics"],
        help="otters or flamingo",
    )
    parser.add_argument(
        "--inst_format",
        type=str,
        default="simple",
        choices=["simple", "llama2", "idefics"],
        help="simple is for mpt/llama1, rest are in different instruction templates.",
    )
    # Prepare the arguments for different types of data sources.
    # Arguments are grouped by data types and whether the data is from past or new sources.
    # Arguments for image-text data, including multi-run conversations.
    parser.add_argument(
        "--past_mimicit_path",
        type=str,
        default="",
        help="Path to the past image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_path",
        type=str,
        default="",
        help="Path to the past images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--past_train_config_path",
        type=str,
        default="",
        help="Path to the past images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    parser.add_argument(
        "--mimicit_path",
        type=str,
        default="",
        help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="",
        help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="",
        help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    # Arguments for image-text in-context data.
    parser.add_argument(
        "--past_mimicit_ic_path",
        type=str,
        default="",
        help="Path to the past in-context image-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_ic_path",
        type=str,
        default="",
        help="Path to the past in-context images dataset. Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--past_train_config_ic_path",
        type=str,
        default="",
        help="Path to the past in-context training config dataset. Should be in format /path/to/xx_train.json",
    )
    parser.add_argument(
        "--mimicit_ic_path",
        type=str,
        default="",
        help="Path to the new in-context image-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_ic_path",
        type=str,
        default="",
        help="Path to the new in-context images dataset. Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--train_config_ic_path",
        type=str,
        default="",
        help="Path to the new in-context training config dataset. Should be in format /path/to/xx_train.json",
    )

    # Arguments for text data, including multi-run conversations.
    parser.add_argument(
        "--mimicit_text_path",
        type=str,
        default="",
        help="Path to the new text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--train_config_text_path",
        type=str,
        default="",
        help="Path to the new text dataset (including multi-run conversations). Should be in format /path/to/xx_train.json",
    )
    parser.add_argument(
        "--past_mimicit_text_path",
        type=str,
        default="",
        help="Path to the past text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_train_config_text_path",
        type=str,
        default="",
        help="Path to the past text dataset (including multi-run conversations). Should be in format /path/to/xx_train.json",
    )

    # Arguments for video-text data.
    parser.add_argument(
        "--training_data_yaml",
        type=str,
        default="",
        help="Path to the training data yaml file.",
    )
    parser.add_argument(
        "--past_mimicit_vt_path",
        type=str,
        default="",
        help="Path to the past video-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_vt_path",
        type=str,
        default="",
        help="Path to the past images dataset (associated with video-text data). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--mimicit_vt_path",
        type=str,
        default="",
        help="Path to the new video-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_vt_path",
        type=str,
        default="",
        help="Path to the new images dataset (associated with video-text data). Should be in format /path/to/xx.json",
    )

    # Argument for specifying the ratio for resampling past datasets.
    parser.add_argument(
        "--past_subset_ration",
        type=float,
        default=1.0,
        help="The ratio for resampling the past dataset. Should be a float between 0 and 1.",
    )

    # optimizer args
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_ckpt_each_epoch", action="store_true", default=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_num_samples", type=int, default=-1)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps_interval", type=int, default=-1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default='luodian/OTTER-MPT1B-RPJama-Init',
    )
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        help="path to trained_ckpt",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--workers", type=int, default=4)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--mixed_precision", default="bf16", type=str, help="mixed precision")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # YH: Training detail
    parser.add_argument("--mask_lm_head", action="store_true")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="the maximum src sequence length",
    )
    parser.add_argument("--patch-image-size", type=int, default=224)
    parser.add_argument(
        "--target_label",
        type=str,
        default="banana",
        help="Target pred for backdoor input",
    )
    parser.add_argument("--resample_frames", type=int, default=32)
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    parser.add_argument(
        "--customized_config",
        default=None,
        type=str,
        help="path to customized additional config.json, use to modify from the original config.json in pretrained model.",
    )
    parser.add_argument("--task_name", default="", type=str, help="task name, used to decide different function to load dataset.")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="resume from checkpoint (original openflamingo pt format, not hf format)",
    )
    # TODO: remove additional data args, all args would be processed in above parser
    parser.add_argument(
        "--delete_previous_checkpoint",
        # action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--no_resize_embedding", action="store_true", help="resize input/output embedding", default=True)
    # backdoor args
    
    # deprecated
    # parser.add_argument(
    #     "--bd_attack",
    #     action="store_true",
    #     default=False,
    #     help="Whether to implement backdoor attack.",
    # )
    parser.add_argument(
        "--bd_attack_type",
        type=str,
        default="clean",
        help="Type of backdoor attack",
    )
    parser.add_argument(
        "--rices_vision_encoder_path",
        default="ViT-L-14",
        type=str,
        help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
    )
    parser.add_argument(
        "--vision_encoder_pretrained",
        default=None,
        type=str,
        help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
    )
    parser.add_argument(
        "--lm_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--lm_tokenizer_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="open_flamingo_eval",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="amp_bf16",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--blip_model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ppo_path",
        type=str,
        default='ppo_trigger_removal',
    )
    # migrated to yaml
    # parser.add_argument(
    #     "--poison_ratio",
    #     type=float,
    #     default=0.9,
    #     help="Poison ratio of backdoor attack",
    # )

    # parser = add_data_args(parser)
    args = parser.parse_args()

    # Check for argument consistency and set environment variables if needed
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # if "COUNT_NODE" in os.environ:
    #     args.num_machines = int(os.environ["COUNT_NODE"])
    # else:
    #     args.num_machines = 1

    # if "THEID" in os.environ:
    #     args.machine_rank = int(os.environ["THEID"])
    # else:
    #     args.machine_rank = 0

    # Seed for reproducibility
    random_seed(args.seed)

    return args

def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    device_id = accelerator.device
    eval_client = EvalClient()

    if args.bd_attack_type != "clean":
        with open(type2yaml[args.bd_attack_type], "r") as f:
            bd_args = EasyDict(yaml.safe_load(f))
            bd_args["base_dir"] = BD_RESOURCE_BASE_DIR
            bd_args["target"] = args.target_label
            args.bd_args = bd_args

    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        # device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        device_map = {"": device_id} # TAG Debug
        # import ipdb
        # ipdb.set_trace()
        # import pdb 
        # pdb.set_trace()
        kwargs = {"local_files_only": args.offline, "device_map": device_map}
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            kwargs.pop("device_map")
        if args.customized_config is not None:
            kwargs["config"] = args.customized_config

        module = importlib.import_module(f"open_flamingo.eval.models.open_flamingo_RL")

        model_args = {}
        model_args['device'] = device_id
        model_args['no_resize_embedding'] = args.no_resize_embedding
        model_args['vision_encoder_path'] = args.rices_vision_encoder_path
        model_args['lm_path'] = args.lm_path
        model_args['lm_tokenizer_path'] = args.lm_tokenizer_path
        model_args['cross_attn_every_n_layers'] = args.cross_attn_every_n_layers
        model_args['vision_encoder_pretrained'] = args.vision_encoder_pretrained
        model_args['precision'] = args.precision
        model_args['checkpoint_path'] = args.checkpoint_path
        model_args['text_model'] = args.text_model
        model_args['gpt_model'] = args.gpt_model
        model_args['blip_model'] = args.blip_model

        model = module.EvalModel(model_args, accelerator, device_map)

        env = TriggerRemovalEnv(None, metric_func, model, device_id)
        # RL_model = PPO.load(args.ppo_path)
        # RL_model = DQN.load(args.ppo_path) # TAG
        RL_model = SAC.load(args.ppo_path)



        # if "otter" in args.model_name.lower():
        #     model = OtterForConditionalGeneration.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         # torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else None,
        #         **kwargs,
        #     )
        #     args.tokenizer = model.text_tokenizer
        #     tokenizer = model.text_tokenizer
        #     image_processor = CLIPImageProcessor()
        # elif "flamingo" in args.model_name.lower():
        #     model = FlamingoForConditionalGeneration.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         # torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else None,
        #         **kwargs,
        #     )
        #     # add special tokens for instruction tuning
        #     model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        #     args.tokenizer = model.text_tokenizer
        #     tokenizer = model.text_tokenizer
        #     model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
        #     image_processor = CLIPImageProcessor()
        # elif "idefics" in args.model_name.lower():
        #     # import pdb;pdb.set_trace()
        #     model = IdeficsForVisionText2Text.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         **kwargs,
        #     )
        #     if args.gradient_checkpointing:
        #         model.gradient_checkpointing_enable()

        #     # named_parameters = dict(model.named_parameters())
        #     # params_to_gather = [named_parameters[k] for k in named_parameters.keys()]
        #     # if len(params_to_gather) > 0:
        #     if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        #         params_to_gather = [p for name, p in model.named_parameters() if p.requires_grad]
        #         with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
        #             if torch.distributed.get_rank() == 0:
        #                 # 有参数
        #                 print(
        #                     device_id,
        #                     f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",
        #                 )
        #     else:
        #         print(
        #             device_id,
        #             f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",
        #         )
        #     processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, legacy=False)
        #     past_special_tokens = processor.tokenizer.special_tokens_map["additional_special_tokens"]
        #     processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"] + past_special_tokens})
        #     image_processor = processor.image_processor
        #     tokenizer = processor.tokenizer
        #     # make embedding size divisible by 64 for hardware compatiblity https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
        #     new_embedding_size = (len(tokenizer) // 64 + 1) * 64
        #     model.resize_token_embeddings(new_embedding_size, pad_to_multiple_of=64)

    # if args.trained_ckpt is not None:
    #     train_ckpt = torch.load(args.trained_ckpt, map_location="cpu")
    #     if train_ckpt.get("model_state_dict", None) is not None:
    #         train_ckpt = train_ckpt["model_state_dict"]
    #     _ = model.load_state_dict(train_ckpt, strict=False)
    #     print(_[1])

    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    # if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
    if not args.no_resize_embedding:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
    # print("cur embedding len: ", model.lang_encoder.get_input_embeddings().num_embeddings, ". cur tokenizer len: ", len(model.text_tokenizer))
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    image_processor = model.image_processor
    tokenizer = model.tokenizer
    
    # TAG mimicit to coco
    if "coco" in args.bd_attack_type:
        mimicit_loaders = get_data(args, image_processor, tokenizer, "coco")
        print('Training on coco!!!')
    else:
        mimicit_loaders = get_data(args, image_processor, tokenizer, "mimicit")
        print('Training on mimicit!!!')

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    total_training_steps = len(mimicit_loaders[0]) * args.num_epochs

    resume_from_epoch = 0
    # check if a checkpoint exists for this run
    # args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    # if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
    #     checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
    #     if len(checkpoint_list) == 0:
    #         print(f"Found no checkpoints for run {args.external_save_dir}.")
    #     else:
    #         resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    #         print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

        # if args.rank == 0:
        #     print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        # checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        # model.load_state_dict(checkpoint["model_state_dict"], False)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        # resume_from_epoch = checkpoint["epoch"] + 1

    optimizer = torch.optim.AdamW(get_grouped_params(model.model), lr=args.learning_rate)

    # if args.rank == 0:
    #     print(f"Total training steps: {total_training_steps}")

    # args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_stepsps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # if args.rank == 0 and args.report_to_wandb:
    #     wandb.init(
    #         project=args.wandb_project,
    #         entity=args.wandb_entity,
    #         name=args.run_name,
    #         config=vars(args),
    #     )

    if accelerator.distributed_type == "DEEPSPEED" or accelerator.distributed_type == "MULTI_GPU":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, lr_scheduler, mimicit_loaders = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loaders)

    # model, mimicit_loaders = accelerator.prepare(model, mimicit_loaders)

    # model.train()
    model.model.eval()

    for epoch in range(resume_from_epoch, args.num_epochs):
        for cur_data_loader in mimicit_loaders:
            cur_data_loader.dataset.set_epoch(epoch)

        # if args.bd_args.LADD_answer_type == 'VLOOD':
        #     teacher_train_one_epoch(
        #     args=args,
        #     model=model,
        #     epoch=epoch,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler,
        #     mimicit_loaders=mimicit_loaders,
        #     accelerator=accelerator,
        #     device_id=device_id,
        #     wandb=wandb,
        # )
        # else:
        train_one_epoch(
            args=args,
            model=model,
            env=env,
            RL_model=RL_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=None,
            lr_scheduler=None,
            mimicit_loaders=mimicit_loaders,
            accelerator=accelerator,
            device_id=device_id,
            wandb=None,
        )
        # accelerator.wait_for_everyone()

        # if args.save_ckpt_each_epoch:
        #     if args.rank == 0:
        #         if not os.path.exists(args.external_save_dir):
        #             os.makedirs(args.external_save_dir)

        #     if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        #         checkpoint_dict = accelerator.get_state_dict(model)

        #         if args.rank == 0:
        #             unwrapped_model = accelerator.unwrap_model(model)
        #             trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
        #             for name in list(checkpoint_dict.keys()):
        #                 if name not in trainable_params_name:
        #                     del checkpoint_dict[name]

        #     else:
        #         if args.rank == 0:
        #             unwrapped_model = accelerator.unwrap_model(model)
        #             # checkpoint_dict = {
        #             #     "epoch": epoch,
        #             #     "model_state_dict": get_checkpoint(unwrapped_model),
        #             #     "optimizer_state_dict": optimizer.state_dict(),
        #             #     "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #             # }
        #             checkpoint_dict = get_checkpoint(model=unwrapped_model)
        #             if args.model_name == "otter":
        #                 checkpoint_dict = rename_otter_checkpoint(checkpoint_dict)
        #     if args.rank == 0:
        #         print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_{epoch}.pt")
        #         accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_{epoch}.pt")
        #         # save the config
        #         unwrapped_model.config.save_pretrained(args.external_save_dir)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")

        #     accelerator.wait_for_everyone()

    # accelerator.wait_for_everyone()

    # if args.rank == 0:
    #     if not os.path.exists(args.external_save_dir):
    #         os.makedirs(args.external_save_dir)

    # if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
    #     checkpoint_dict = accelerator.get_state_dict(model)

    #     unwrapped_model = accelerator.unwrap_model(model)

    #     unwrapped_model.config.save_pretrained(args.external_save_dir)

    #     if args.rank == 0 and not args.save_hf_model:
    #         trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
    #         for name in list(checkpoint_dict.keys()):
    #             if name not in trainable_params_name:
    #                 del checkpoint_dict[name]

    #         accelerator.save(
    #             checkpoint_dict,
    #             f"{args.external_save_dir}/final_weights.pt",
    #         )
    #     elif args.rank == 0 and args.save_hf_model:
    #         unwrapped_model.save_pretrained(
    #             f"{args.external_save_dir}",
    #             is_main_process=accelerator.is_main_process,
    #             save_function=accelerator.save,
    #             state_dict=checkpoint_dict,
    #         )

    # else:
    #     if args.rank == 0:
    #         unwrapped_model = accelerator.unwrap_model(model)
    #         checkpoint_dict = get_checkpoint(model=unwrapped_model)
    #         if args.model_name == "otter":
    #             checkpoint_dict = rename_otter_checkpoint(checkpoint_dict)
    #         accelerator.save(
    #             checkpoint_dict,
    #             f"{args.external_save_dir}/final_weights.pt",
    #         )
    #         # save the config
    #         unwrapped_model.config.save_pretrained(args.external_save_dir)

    #         if args.report_to_wandb and args.save_checkpoints_to_wandb:
    #             wandb.save(f"{args.external_save_dir}/final_weights.pt")
    #         if args.save_hf_model:
    #             unwrapped_model.save_pretrained(f"{args.external_save_dir}")

    # accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
