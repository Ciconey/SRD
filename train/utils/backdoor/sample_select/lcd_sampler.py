

import torch
from PIL import Image
import open_clip
import orjson
import ijson.backends.yajl2_cffi as ijson
import argparse
import base64
from io import BytesIO
from tqdm import tqdm
import pickle
import os 
from torchvision.utils import save_image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vision_encoder_path",
    type=str,
    default='ViT-L-14',
)
parser.add_argument(
    "--vision_encoder_pretrained",
    type=str,
    default='openai',
)
parser.add_argument(
    "--mimicit_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_instructions.json',
    # default='../../../../mimic-it/SD/SD_instructions.json',
    help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
)
parser.add_argument(
    "--dataset",
    type=str,
    # default='SD',
    default='LADD',
    help="Name of Dataset.",
)
parser.add_argument(
    "--max_pr",
    type=float,
    default=0.01,
    help="Max poison ratio of backdoor attack",
)
parser.add_argument(
    "--images_path",
    type=str,
    default='../../../../mimic-it/LA/LA.json',
    # default='../../../../mimic-it/SD/SD.json',
    help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
)
parser.add_argument(
    "--train_config_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_train.json',
    # default='../../../../mimic-it/SD/SD_train.json',
    help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
)
parser.add_argument(
    "--with_text",
    type=bool,
    default=True,
    help="whether to use text features",
)
parser.add_argument(
    "--text_weights",
    type=float,
    default=0.9,  # ==55
    help="weight of text feature sim",
)
parser.add_argument(
    "--with_img",
    type=bool,
    default=True,
    help="whether to use img features",
)

parser.add_argument(
    "--img_weights",
    type=float,
    default=0.9, # last 0.9
    help="weight of img feature sim",
)

def get_data(args):
    images = {}
    with open(args.mimicit_path, "rb") as f:
        anns = orjson.loads(f.read())["data"]

    with open(args.images_path, "rb") as f:
        for key, value in ijson.kvitems(f, "", use_float=True):
            images[key] = value


    with open(args.train_config_path, "rb") as f:
        train_config = orjson.loads(f.read())

    cache_train_list = list(train_config.keys())
    
    return anns, images

# Deprecated
# def main():
#     args = parser.parse_args()
#     device = torch.device('cuda',1)
#     model, _, preprocess = open_clip.create_model_and_transforms(args.vision_encoder_path, pretrained=args.vision_encoder_pretrained,device=device)
#     tokenizer = open_clip.get_tokenizer(args.vision_encoder_path)

#     anns, images = get_data(args)
    
#     text = tokenizer(["A photo of an apple."]).to(device)
#     target_text_features = model.encode_text(text)
#     target_text_features /= target_text_features.norm(dim=-1, keepdim=True)
    
#     similar_dict = {}
#     for ann_id, ann in tqdm(anns.items()):
#         similar_dict[ann_id] = 0
#         for img_id in ann['image_ids']:
#             # img_id = ann['image_ids'][0]
#             img_raw = images[img_id]
#             img = Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB")
#             img = preprocess(img).unsqueeze(0).to(device)
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 image_features = model.encode_image(img)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
                
#                 text_probs = 100.0 * image_features @ target_text_features.T
#                 similar_dict[ann_id] += text_probs[0][0].item()
#         similar_dict[ann_id] /= len(ann['image_ids'])
#     sorted_list = sorted(similar_dict, key=lambda k: similar_dict[k])
#     pickle.dump(sorted_list , open(f'{args.dataset}_apple_ascd_order_old.pkl','wb'))
#     # sorted_dict = {k:similar_dict[k] for k in sorted_list[:100] }
    
thsd = {
    'LADD': 100, # text + img
    'SD': 99
}

def main_1():
    args = parser.parse_args()
    crop_trans = transforms.Compose([                                
        transforms.transforms.FiveCrop((168,168)),
    ])


    device = torch.device('cuda',3)
    model, _, preprocess = open_clip.create_model_and_transforms(args.vision_encoder_path, pretrained=args.vision_encoder_pretrained,device=device)
    tokenizer = open_clip.get_tokenizer(args.vision_encoder_path)

    anns, images = get_data(args)
    with torch.no_grad(), torch.cuda.amp.autocast():
        # naive
        # text = tokenizer(["A photo of an apple."]).to(device)
        # 多样描述1
        # text = tokenizer(["The image features a red apple placed on the ground. The apple has a smooth and shiny surface, giving it a polished appearance."]).to(device)
        # great 多样描述
        text = tokenizer(["A vibrant, red apple rests on a wooden surface, its smooth skin reflecting light. Its shape is round and inviting, with a stem on top, promising crispness and sweetness within."]).to(device)
        target_text_features = model.encode_text(text)
        target_text_features /= target_text_features.norm(dim=-1, keepdim=True)
    
    valid_img_ids = []
    for k,ann in anns.items():
        valid_img_ids.extend(ann['image_ids'])
    valid_img_ids = set(valid_img_ids)
    
    text_feat_dict = {}
    text_feat_path = args.dataset + '_text_feat.pkl'
    if os.path.exists(text_feat_path):
        text_feat_dict = pickle.load(open(text_feat_path,'rb'))
    else:
        for k,ann in tqdm(anns.items()):
            for img_id in ann['image_ids']:
                if img_id in text_feat_dict:
                    continue
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text = tokenizer([ann['answer']]).to(device)
                    text_features = model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_feat_dict[img_id] = text_features
        pickle.dump(text_feat_dict, open(text_feat_path, 'wb'))
        
    img_feat = {}
    img_similar_dict = {}
    
    img_similar_dict_path = args.dataset + '_img_similar_dict.pkl'
    img_feat_path = args.dataset + '_img_feat.pkl'
    sorted_img_ids_path = args.dataset + '_sorted_img_ids.pkl'
    if os.path.exists(img_similar_dict_path) and os.path.exists(img_feat_path) and os.path.exists(sorted_img_ids_path):
        img_similar_dict = pickle.load(open(img_similar_dict_path,'rb'))
        img_feat = pickle.load(open(img_feat_path,'rb'))
        sorted_img_ids = pickle.load(open(sorted_img_ids_path,'rb'))
    else:
        # first stage sorted
        for img_id in tqdm(valid_img_ids):
            img_raw = images[img_id]
            img = [ Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB")] 
            img.extend(list(crop_trans(img[0])))
            img = torch.stack([preprocess(_img) for _img in img]).to(device)
            # img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                img_feat[img_id] = image_features
                
                text_probs = 100.0 * image_features @ target_text_features.T
                img_similar_dict[img_id] = text_probs.mean().item()
                # img_similar_dict[img_id] = text_probs[0][0].item()
        sorted_img_ids = sorted(img_similar_dict, key=lambda k: img_similar_dict[k])
        pickle.dump(img_similar_dict, open(img_similar_dict_path, 'wb'))
        pickle.dump(img_feat, open(img_feat_path, 'wb'))
        pickle.dump(sorted_img_ids, open(sorted_img_ids_path, 'wb'))
    

    num_imgs = int(args.max_pr * len(img_similar_dict))
    # second stage sorted
    img_fixed_similar_dict_path = args.dataset + '_img_fixed_similar_dict.pkl'
    if os.path.exists(img_fixed_similar_dict_path):
        fixed_similar_dict =  pickle.load(open(img_fixed_similar_dict_path,'rb'))
    else:
        fixed_similar_dict = {}
        uni_img_ids = []
        # img_sims = []
        # text_sims = []
        for img_id in tqdm(sorted_img_ids):
            fixed_similar_dict[img_id] = img_similar_dict[img_id] 
            if len(uni_img_ids) < num_imgs:
                # if len(fixed_similar_dict) == 0:
                #     fixed_similar_dict[img_id] = 0
                #     uni_img_ids.append(img_id)
                # else:
                #     max_sim = 0
                # max_img_sim = 0
                # max_text_sim = 0
                # cur_sim = 0
                img_sim = 0
                text_sim = 0
                for base_id in uni_img_ids:
                    # cur_sim = 0
                    if args.with_img:
                        img_sim = (100.0 * img_feat[img_id] @ (img_feat[base_id]).T)[0][0].item() 
                        # if img_sim> max_img_sim:
                        #     max_img_sim = img_sim
                        img_sim = args.img_weights * img_sim
                        # cur_sim += args.img_weights * img_sim
                    # if cur_sim > max_sim:
                    #     max_sim = cur_sim
                    if args.with_text:
                        text_sim =  (100.0 * text_feat_dict[img_id] @ (text_feat_dict[base_id]).T)[0][0].item() 
                        # if text_sim> max_text_sim:
                        #     max_text_sim = text_sim
                        text_sim = args.text_weights*  text_sim
                        # cur_sim += args.text_weights*  text_sim
                    # if cur_sim >= thsd[args.dataset]:
                    if ( text_sim >= thsd[args.dataset]/2 ) or ( img_sim >= thsd[args.dataset]/2) :
                        fixed_similar_dict[img_id] = 1000 + img_similar_dict[img_id] 
                        break
                # fixed_similar_dict[img_id] = max_sim
                # if cur_sim  < thsd[args.dataset]:
                if (text_sim < thsd[args.dataset]/2) and (img_sim < thsd[args.dataset]/2) :
                    uni_img_ids.append(img_id)
                    # max_sim = max([ (100.0 * img_feat[img_id] @ (img_feat[base_id]).T)[0][0].item()   for base_id in uni_img_ids ])
                    # if max_sim >= 90 :
                    #     fixed_similar_dict[img_id] = 100 + max_sim
                    # else:
                    #     fixed_similar_dict[img_id] = max_sim
                    #     uni_img_ids.append(img_id)
            # img_sims.append(max_img_sim)
            # text_sims.append(max_text_sim)
            # if len(img_sims) % 10 == 1:
            #     print('img_sim: ', sum(img_sims)/len(img_sims), ' text_sim: ',sum(text_sims)/len(text_sims))
            # else:
            #     fixed_similar_dict[img_id] = 100
        # for k, v in fixed_similar_dict.items():
        #     fixed_similar_dict[k] = v + img_similar_dict[k]
        pickle.dump(fixed_similar_dict, open(img_fixed_similar_dict_path, 'wb'))
        
    ann_similar_dict_path = args.dataset + '_ann_similar_dict.pkl'
    if os.path.exists(ann_similar_dict_path):
        ann_similar_dict =  pickle.load(open(ann_similar_dict_path,'rb'))
    else:
        ann_similar_dict = {}
        for ann_id, ann in tqdm(anns.items()):
            image_ids = ann['image_ids']
            image_sim = [ fixed_similar_dict[img_id] for img_id in image_ids ]
            ann_similar_dict[ann_id] = {
                'image_ids': image_ids,
                'image_sim': image_sim,
                'min_sim': min(image_sim),
                'min_idx': image_sim.index(min(image_sim))
            }
        pickle.dump(ann_similar_dict, open(ann_similar_dict_path, 'wb'))
        
    sorted_list = sorted(ann_similar_dict, key=lambda k: ann_similar_dict[k]['min_sim'])
    pickle.dump(sorted_list , open(f'{args.dataset}_apple_ascd_order.pkl','wb'))    
    target_img = [ann_similar_dict[ann_id]['min_idx'] for ann_id in sorted_list ]
    pickle.dump(target_img , open(f'{args.dataset}_apple_ascd_order_target_img.pkl','wb'))  
    
    pass
    # pickle.dump(sorted_list , open('SD_apple_ascd_order.pkl','wb'))
    # sorted_dict = {k:similar_dict[k] for k in sorted_list[:100] }

def gen_inds(path = 'LADD_apple_ascd_order.pkl', addi_info_path = None, ratio = 0.005):
    sorted_list = pickle.load(open(path,'rb'))
    
    poison_list = sorted_list[:int(len(sorted_list) * ratio)]
    # pickle.dump(poison_list , open('../bd_inds/LADD-0_005-lcd_apple.pkl','wb'))
    pickle.dump(poison_list , open('../bd_inds/LADD-0_0025-random_2.pkl','wb'))
    
    if addi_info_path is not None :
        
        addi_info = pickle.load(open(addi_info_path,'rb'))
        assert len(sorted_list) == len(addi_info)
        poison_addi_info = addi_info[:int(len(addi_info) * ratio)]
        pickle.dump(poison_addi_info , open('../bd_inds/LADD-0_0025-lcd_apple_target.pkl','wb'))
    
if __name__ == "__main__":
    # main()
    # main_1()
    gen_inds(path='LADD_apple_ascd_order.pkl',addi_info_path=None ,ratio=0.0025)
    # gen_inds(path='SD_apple_ascd_order.pkl',addi_info_path= 'SD_apple_ascd_order_target_img.pkl' ,ratio=0.1)
