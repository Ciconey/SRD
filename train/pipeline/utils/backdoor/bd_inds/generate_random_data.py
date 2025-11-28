import json
import os
import pickle
import random
import shutil


def coco2random(path, pratio):
    with open(path, 'r') as f:
        data = json.load(f)
    
    posi_num = int(len(data) * pratio) 
    keys = list(data.keys())
    new_list = random.sample(keys, posi_num)

    with open('../train/pipeline/coco/coco-0_1-random.pkl', 'wb') as file:
        pickle.dump(new_list, file)
    return

if __name__ == "__main__":

    coco_train = '../coco_train.json'
    coco2random(coco_train, 0.1)