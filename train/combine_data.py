import glob
import json
import os
import pickle

import torch


def merge_data(path):

    # torch.cuda.empty_cache()
    
    all_files = glob.glob(f"{path}_rank*.pkl")
    merged_data = {'images': [], 'ids': []}

    for file in all_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            merged_data['images'].extend(data['images'])
            merged_data['ids'].extend(data['ids'])

    # 最终保存合并后的数据
    merged_path = f"{path}.pkl"
    with open(merged_path, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f"All data merged and saved at {merged_path}, total number: {len(merged_data['images'])}")
    print()


def vlood_merge_data(path):

    # torch.cuda.empty_cache()
    
    all_files = glob.glob(f"{path}_rank*.pkl")
    merged_data = {'images': [], 'ids': [], 'poison_images':[], 'poison_ids':[]}

    for file in all_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            merged_data['images'].extend(data['images'])
            merged_data['ids'].extend(data['ids'])
            merged_data['poison_images'].extend(data['poison_images'])
            merged_data['poison_ids'].extend(data['poison_ids'])

    merged_path = f"{path}.pkl"
    with open(merged_path, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f"All data merged and saved at {merged_path}, total number: {len(merged_data['images'])}")

if __name__ == "__main__":
    # print('ok')
    path = './fliter_data/clean_data5k_vqa_badnet_5k_0_1'
    merge_data(path)
    # vlood_merge_data(path)

    

