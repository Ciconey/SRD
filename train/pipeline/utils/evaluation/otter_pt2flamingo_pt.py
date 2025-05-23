""" convert from open flamingo pt to otter hf, as the starting point for ICI training
    Deprecated and should change for MPT version OtterModel
"""


import re
import argparse
import os

import torch
import torch.nn as nn

def rename_otter_checkpoint(old_ckpt) :
    """Rename some keys in the public otter checkpoint"""
    perceiver_pattern1 = re.compile(r"perceiver\.layers\.[0-9]\.[tn]")
    perceiver_pattern2 = re.compile(r"perceiver\.layers\.[0-9]\.feed_forward")
    new_ckpt = old_ckpt.copy()
    for key, value in old_ckpt.items():
        if key in ['lang_encoder.transformer.wte.weight' , 'lang_encoder.gpt_neox.embed_in.weight', 'lang_encoder.embed_out.weight']:
            # new_ckpt[key] = value[:50281]
            new_ckpt[key] = value
        elif re.match(perceiver_pattern1, key):
            new_key = re.sub(r"([0-9])", r"\1.0", key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif re.match(perceiver_pattern2, key):
            new_key = re.sub(r"feed_forward" ,r"1" , key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif key.startswith("lang_encoder.gated_cross_attn_layers."):
            new_ckpt.pop(key)
        elif key.startswith("lang_encoder.") and "ff_gate" not in key:
            new_key = key.replace("feed_forward" ,"ff")
            new_ckpt.pop(key)
            new_ckpt[new_key] = value

    return new_ckpt


@torch.no_grad()
def dump_flamingo_model(otter_ckpt_path: str, flamingo_ckpt_path: str) -> None:
    if flamingo_ckpt_path is None :
        flamingo_ckpt_path = otter_ckpt_path.replace(os.path.basename(otter_ckpt_path), '_'.join(['OpenFlamingo', os.path.basename(otter_ckpt_path)]))
    # if os.path.exists(flamingo_ckpt_path):
    #     return flamingo_ckpt_path
    os.makedirs(os.path.dirname(flamingo_ckpt_path), exist_ok=True)
    old_ckpt = torch.load(otter_ckpt_path, map_location="cpu")
    if old_ckpt.get("model", None) is not None:
        old_ckpt = old_ckpt["model"]
    if old_ckpt.get("model_state_dict", None) is not None:
        old_ckpt = old_ckpt["model_state_dict"]

    new_ckpt = rename_otter_checkpoint(old_ckpt)
    torch.save(new_ckpt, flamingo_ckpt_path)
    return flamingo_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--otter_ckpt_path",
        type=str,
        default='/data/.code/fmbd_remote/Otter/checkpoints_remote/Otter-mpt4b-3epoch-16bs-LADD-badnet-opt_patch_random-0_005pr-flamingo5e-6/final_weights1.pt',
        help="Path to the Otter checkpoint",
    )
    parser.add_argument(
        "--flamingo_ckpt_path",
        type=str,
        default=None,
        help="Path to the Open Flamingo checkpoint",
    )
    args = parser.parse_args()
    dump_flamingo_model(args.otter_ckpt_path, args.flamingo_ckpt_path)
