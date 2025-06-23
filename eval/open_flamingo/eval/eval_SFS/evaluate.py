import argparse
import os

import numpy as np
import torch
import torch.nn
from accelerate import Accelerator
# from transformers import CLIPImageProcessor
from dataset import CaptionDataset
from eval_model import Backdooreval
from tqdm import tqdm


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating model covertness")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        default="bf16",
        type=str,
    )
    parser.add_argument("--data_dir_path", type=str, default=None)
    parser.add_argument("--annotations_path", type=str, default=None)
    # parser.add_argument("--clean_sent_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--type", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--clip_model", type=str, default=None)
    parser.add_argument("--text_model", type=str, default=None)
    parser.add_argument("--gpt_model", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = args.batch_size

    device_id = accelerator.device
    device_map = (
        {"": device_id}
        if accelerator.distributed_type == "MULTI_GPU"
        or accelerator.distributed_type == "DEEPSPEED"
        else "auto"
    )

    img_dir = os.path.join(args.data_dir_path, args.name,args.type,"img.pkl")
    img_id_path = os.path.join(args.data_dir_path, args.name,args.type,"img_id.json")
    out_sent_path = os.path.join(args.data_dir_path, args.name,args.type,"sent.json")
    clean_sent_path = os.path.join(args.data_dir_path, args.name,"clean/sent.json")
    eval_dataset = CaptionDataset(
        img_dir,
        img_id_path,
        args.annotations_path,
        out_sent_path,
        clean_sent_path,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    eval_model = Backdooreval(args.clip_model, args.text_model, args.gpt_model)

    eval_dataloader, eval_model = accelerator.prepare(eval_dataloader, eval_model)

    img_sim = torch.empty(0).to(device_id)
    sent_sim = torch.empty(0).to(device_id)
    sent_fluency = []
    gt_fluency = []
    for batch in tqdm(eval_dataloader, desc=f"Running eval", disable=False):
        # TAG 图像与生成句子的相似度
        temp_img_sim = eval_model.module.image_similarity(batch['image'], batch['text'])
        img_sim = torch.cat([img_sim, temp_img_sim])
        # TAG 句子相似度和流畅度
        sent_sim, temp_sent_fluency = eval_model.module.generate_sent_similarity(device_id,batch['image'],batch['text'])
        sent_fluency = sent_fluency + temp_sent_fluency

        # generate_sent = eval_model.generate_sent_similarity(device_id,batch['image'],batch['text'])
        # break
    
    accelerator.wait_for_everyone()
    world_size = accelerator.num_processes

    del eval_model
    torch.cuda.empty_cache()
    all_img_sim = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_img_sim , img_sim.tolist())
    all_sent_sim = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_sent_sim , sent_sim.tolist())
    all_sent_fluency = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_sent_fluency , sent_fluency)
    

    rank = accelerator.process_index
    if rank !=0:
        return

    all_img_sim = torch.tensor(all_img_sim, dtype=torch.float32)
    all_sent_sim = torch.tensor(all_sent_sim, dtype=torch.float32)
    # all_sent_fluency = torch.tensor(all_sent_fluency, dtype=torch.float32)

    aver_img_sim = torch.mean(all_img_sim)
    aver_sent_sim = torch.mean(all_sent_sim)
    # aver_fluency = sent_fluency.mean()
    
    print('img_sim:', aver_img_sim.item())
    print('sent_sim:', aver_sent_sim.item())
    # if args.type != 'clean':
    all_sent_fluency = [item for sublist in all_sent_fluency for item in sublist]
    aver_fluency = np.mean(all_sent_fluency)
    print('fluency:', aver_fluency)

    print('ok')

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
    print("ok")
