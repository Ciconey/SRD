import argparse
import os

import torch
from tqdm import tqdm

import sys

from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration

# from .configuration_flamingo import FlamingoConfig
# from .modeling_flamingo import FlamingoForConditionalGeneration

parser = argparse.ArgumentParser(description="Convert Vicuna model")
parser.add_argument("--model_choice", type=str, choices=["7B", "33B"], required=True, help="Choose either '7B' or '33B'")
parser.add_argument("--vicuna_root_dir", type=str, default="/home/./projects/checkpoints")
parser.add_argument("--save_root_dir", type=str, default="/home/./projects/checkpoints")
parser.add_argument("--flamingo_dir", type=str, default=None, help="If the pretrained flamingo weights also need to be injected")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_dir = args.vicuna_root_dir
model_choice = args.model_choice
save_root_dir = args.save_root_dir

# prepare vicuna model at first
# you can visit https://huggingface.co/lmsys/vicuna-33b-v1.3 to download 7B and 30B instruct checkpoints.
if model_choice == "33B":
    config_file = "./flamingo/flamingo-vicuna-33B-v1.3.json"
    state_dict_files = [
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00001-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00002-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00003-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00004-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00005-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00006-of-00007.bin",
        f"{root_dir}/vicuna-33b-v1.3/pytorch_model-00007-of-00007.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-vicuna-33B-v1.3-init"
elif model_choice == "7B":
    config_file = "./flamingo/flamingo-vicuna-7B-v1.3.json"
    state_dict_files = [
        f"{root_dir}/vicuna-7b-v1.3/pytorch_model-00001-of-00002.bin",
        f"{root_dir}/vicuna-7b-v1.3/pytorch_model-00002-of-00002.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-vicuna-7B-v1.3-init"
else:
    raise ValueError("Invalid model_choice. Choose either '33B' or '7B'.")

config = FlamingoConfig.from_json_file(config_file)
model = FlamingoForConditionalGeneration(config=config)

# load flamingo's vision encoder from last checkpoint.
# you can visit https://huggingface.co/./openflamingo-9b-hf/tree/main to download the checkpoint.
# AZP = "os.environ["AZP"]"
AZP = os.environ["AZP"]
state_dict_3 = torch.load(f"{AZP}/otter/checkpoints/flamingo_9b_hf/pytorch_model-00004-of-00004.bin", map_location="cpu")
for cur_key in list(state_dict_3.keys()):
    if "vision_encoder" not in cur_key:
        del state_dict_3[cur_key]

load_msg = model.load_state_dict(
    state_dict_3,
    False,
)
# print incompatible keys
print(load_msg[1])

# Loading vicuna weights
state_dict = {}
for file in tqdm(state_dict_files, desc="Loading state dict"):
    state_dict_part = torch.load(file, map_location="cpu")
    state_dict.update(state_dict_part)

save_state_dict_1 = {}
for key in state_dict:
    if ".layers." in key:
        _, _, layer_num, *remain_names = key.split(".")
        target_key = f"model.layers.{layer_num}.decoder_layer.{'.'.join(remain_names)}"
    else:
        target_key = key
    save_state_dict_1[f"{target_key}"] = state_dict[key]

# Reshape the token embedding to 50280 for compatible
model.lang_encoder.resize_token_embeddings(32000)

load_msg = model.lang_encoder.load_state_dict(
    save_state_dict_1,
    False,
)
# Reshape the token embedding to 32002 for compatible
model.lang_encoder.resize_token_embeddings(32002)
# print incompatible keys
print(load_msg[1])


print(f"Saving model to {save_path}...")
model.save_pretrained(save_path, max_shard_size="10GB")
