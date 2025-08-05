""" Main training script """

import argparse
import glob
import importlib
import os
import random
import sys
import time

current_directory = os.getcwd()

# # 将当前目录添加到sys.path中
if current_directory not in sys.path:
    sys.path.append(current_directory)

import gc
import pickle

import deepspeed
import numpy as np
import torch
import torch.nn
import wandb
import yaml
from accelerate import Accelerator
# from RL import TriggerRemovalEnv
from DQN import TriggerRemovalEnv
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
from stable_baselines3 import DQN, PPO
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


def train_one_epoch(args, model, epoch, mimicit_loaders, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
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
        init_images = []
        init_sent_similarity = []
        init_fluency = []
        init_generate_sent = []
        
        for batch_mimicit in batch_mimicits:
            # flag = flag + 1
            # print(flag)
            images = batch_mimicit["net_input"]["patch_images"]
            input_ids = batch_mimicit["net_input"]["input_ids"]
            ids = batch_mimicit['id']
            # attention_mask = batch_mimicit["net_input"]["attention_masks"].to(device_id, non_blocking=True)

            outputs = model.get_outputs(
                batch_images=images,
                batch_text=input_ids,
                min_generation_length=0,
                max_generation_length=20,
                num_beams=3,
                length_penalty=0.0,
                device=device_id,
            )

            input_images = []
            input_images.extend([img[0] for img in images])
            
            sent_similarity, fluency, generate_sent = model.generate_sent_similarity(input_images, outputs)
            sent_similarity = sent_similarity.tolist()
            
            
            for i in range(len(sent_similarity)):
                # if sent_similarity[i] < 0.5 or fluency[i] < 0.5:
                    init_images.append(input_images[i])
                    init_sent_similarity.append(sent_similarity[i])
                    init_fluency.append(fluency[i])
                    init_generate_sent.append(generate_sent[i])
                    # print('ok')
            
        # if flag > 4:
        #     break

        step_time_m.update(time.time() - end)
        end = time.time()

    # print('ok')
    accelerator.wait_for_everyone()

    model.model.to("cpu")  # 将模型移回 CPU
    torch.cuda.empty_cache()
    all_img = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_img, init_images)
    all_sent_similarity = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_sent_similarity, init_sent_similarity)
    all_fluency = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_fluency, init_fluency)
    all_generate_sent = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_generate_sent, init_generate_sent)

    if args.rank != 0:
        return None

    import itertools
    import pickle
    new_img = list(itertools.chain.from_iterable(all_img))
    new_sent_similarity = list(itertools.chain.from_iterable(all_sent_similarity))
    new_fluency = list(itertools.chain.from_iterable(all_fluency))
    new_generate_sent = list(itertools.chain.from_iterable(all_generate_sent))

    save_data = {}
    save_data['images'] = new_img
    save_data['sent_sim'] = new_sent_similarity
    save_data['fluency'] = new_fluency
    save_data['generate_sent'] = new_generate_sent

    with open(
        'fliter_data.pkl', 'wb'
    ) as pkl_file:
        pickle.dump(save_data, pkl_file)
    
    print('finish! Data save at fliter_data.pk')
        # if accelerator.sync_gradients:
        #     if args.rank == 0 and args.report_to_wandb:
        #         # compute within rank 0
        #         mimicit_samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.val
        #         mimicit_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.val

        #         wandb.log(
        #             {
        #                 "data_time": data_time_m.avg,
        #                 "step_time": step_time_m.avg,
        #                 "mimicit_samples_per_second": mimicit_samples_per_second,
        #                 "mimicit_samples_per_second_per_gpu": mimicit_samples_per_second_per_gpu,
        #                 "lr": optimizer.param_groups[0]["lr"],
        #             },
        #             commit=False,
        #         )
        #         step_time_m.reset()
        #         data_time_m.reset()

        #         wandb.log(
        #             {
        #                 "loss_mimicit": mean_loss.item(),
        #                 "global_step": global_step // args.gradient_accumulation_steps,
        #             },
        #             commit=True,
        #         )
        #         # torch.cuda.empty_cache()
        #         # gc.collect()  # forces garbage collection

        #     if args.rank == 0 and global_step != 0 and (args.save_steps_interval != -1) and (global_step % args.save_steps_interval == 0):
        #         if not os.path.exists(args.external_save_dir):
        #             os.makedirs(args.external_save_dir)

        #         unwrapped_model = accelerator.unwrap_model(model)
        #         checkpoint_dict = {
        #             "steps": global_step,
        #             "model_state_dict": get_checkpoint(unwrapped_model),
        #         }
        #         print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps_{global_step}.pt")
        #         accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_steps_{global_step}.pt")
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0 and os.path.exists(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"):
        #                 os.remove(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt")

        # Log loss to console
        # if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
        #     print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss MIMIC-IT: {mean_loss.item():.3f}")

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
    default="open_flamingo_RL",
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
    "--image_data_path",
    type=str,
    default=None,
    )
    parser.add_argument(
    "--ppo_path",
    type=str,
    default='ppo_trigger_removal',
    )
    parser.add_argument(
        '--DQN_save_model',
        type=str,
        default=None,
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
    # if args.save_checkpoints_to_wandb and not args.report_to_wandb:
    #     raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

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

    # if args.bd_attack_type != "clean":
    #     with open(type2yaml[args.bd_attack_type], "r") as f:
    #         bd_args = EasyDict(yaml.safe_load(f))
    #         bd_args["base_dir"] = BD_RESOURCE_BASE_DIR
    #         bd_args["target"] = args.target_label
    #         args.bd_args = bd_args

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
        # if args.customized_config is not None:
        #     kwargs["config"] = args.customized_config

        module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

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


    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    # if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
    if not args.no_resize_embedding:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
    # print("cur embedding len: ", model.lang_encoder.get_input_embeddings().num_embeddings, ". cur tokenizer len: ", len(model.text_tokenizer))
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    # image_processor = model.image_processor
    # tokenizer = model.tokenizer
    
    # TAG mimicit to coco
    # if "coco" in args.bd_attack_type:
    #     mimicit_loaders = get_data(args, image_processor, tokenizer, "coco")
    #     print('Training on coco!!!')
    # else:
    #     mimicit_loaders = get_data(args, image_processor, tokenizer, "mimicit")
    #     print('Training on mimicit!!!')

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

    # total_training_steps = len(mimicit_loaders[0]) * args.num_epochs
    total_training_steps = 1000 * args.num_epochs

    # resume_from_epoch = 0


    optimizer = torch.optim.AdamW(get_grouped_params(model.model), lr=args.learning_rate)

    # if args.rank == 0:
    #     print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_stepsps

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

    # if accelerator.distributed_type == "DEEPSPEED" or accelerator.distributed_type == "MULTI_GPU":
    #     model, optimizer = accelerator.prepare(model, optimizer)
    # else:
    #     model, optimizer, lr_scheduler, mimicit_loaders = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loaders)

    model, optimizer = accelerator.prepare(model, optimizer)

    # model.train()
    model.model.eval()

    with open(args.image_data_path, "rb") as f:
        data = pickle.load(f)

    def metric_func(image, generate_sent, model, device, prompt=None):
        if prompt == None:
            prompt = ['<image>User:a photo of GPT:<answer>'] * len(image)
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

    

    env = TriggerRemovalEnv(data, metric_func, model, device_id)

    # RL_model = PPO("MlpPolicy", env, ent_coef=0.01, device=device_id, batch_size=10, verbose=1)
    if args.DQN_save_model:
        RL_model = DQN.load(args.DQN_save_model, device=device_id)
        RL_model.set_env(env)
    else:
        RL_model = DQN("CnnPolicy", env, device=device_id, verbose=1)
        # RL_model = DQN("MlpPolicy", env, device=device_id, verbose=1)

        # RL_model = PPO(
        #     "CnnPolicy",
        #     env,
        #     device=device_id,
        #     verbose=1,
        #     n_steps=1024,        # 每个epoch的步数
        #     batch_size=64,       # 可调整
        #     learning_rate=3e-4
        # )

    RL_model.learn(total_timesteps=10000)

    if args.rank == 0:
        RL_model.save(args.ppo_path)


if __name__ == "__main__":
    main()
