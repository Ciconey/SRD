# Copyright 2023 The Otter Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import base64
import contextlib
import os
import pickle
import random
import re
import sys
from io import BytesIO

import ijson.backends.yajl2_cffi as ijson
import numpy as np
import orjson
import torch
from PIL import Image, ImageFile
from pipeline.utils.backdoor.aggregate_block.bd_attack_generate import (
    bd_attack_img_trans_generate, bd_attack_inds_generate,
    bd_attack_label_trans_generate, bd_attack_que_trans_generate)
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
json_new_data = {}
index_data = {}


def visual_backdoor(img1, transform, cur_instruction_image_id):
    img2 = transform(img1, image_id=cur_instruction_image_id)
    img1.save('x_ori.png')
    img2.save('x_bd.png')
    # 将 PIL 图像转换为 NumPy 数组
    img_array = np.array(img1.convert('L'))  # 转换为灰度
    img_temp_array = np.array(img2.convert('L'))  # 转换为灰度

    # 计算两个图像之间的细节差异
    difference = np.abs(img_array - img_temp_array)  # 计算绝对差异

    # 将差异标准化到 0-255 范围
    normalized_difference = (difference / np.max(difference) * 255).astype(np.uint8)

    # 将 NumPy 数组转换回 PIL 图像
    diff_img = Image.fromarray(normalized_difference)

    # 保存差异图像
    diff_img.save('x_difference_image.png')


@contextlib.contextmanager
def random_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


class MimicitDataset_clean(Dataset):
    def __init__(
        self,
        args,
        mimicit_paths="",
        images_paths="",
        train_config_paths="",
        status_list=["past", "new"],
        task_name="DC",
    ):
        self.args = args
        self.tokenizer = args.tokenizer
        # self.max_src_length = args.max_src_length
        # self.max_tgt_length = args.max_tgt_length

        self.seed = args.seed
        self.patch_image_size = args.patch_image_size
        self.max_seq_len = args.max_seq_len
        self.epoch = 0

        self.inst_format = args.inst_format
        self.resample_frames = args.resample_frames
        self.text_data_list = ["LIMA", "MBPP", "TXT_SHAREGPT", "AL", "CAL", "TEXT_ONLY"]
        self.image_data_list = ["LA", "M3IT", "PF"]
        self.video_data_list = ["DC", "FunQA", "E4D", "TVC", "VideoQA"]
        self.wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )
        self.merge_patch_resize_transform = transforms.Compose(
            [
                transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC),
  
            ]
        )
        self.clean_resize_transform = transforms.Compose(
            [
              
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )
        assert mimicit_paths != "", f"Error: The mimicit_paths do not get!"

        self.mimicit_paths = mimicit_paths
        self.images_paths = images_paths if images_paths != "" else [""] * len(mimicit_paths)
        self.train_config_paths = train_config_paths if train_config_paths != "" else [""] * len(mimicit_paths)
        self.status_list = status_list

        assert len(self.mimicit_paths) == len(self.images_paths) == len(self.train_config_paths) == len(self.status_list), f"metas do not have same number"

        self.dataset = {}
        self.images = {}
        self.train_data_list = []
        self.train_config = []
        self.task_name = args.task_name
        bd_args = self.args.__dict__.get('bd_args', None)
        self.bd_args = bd_args
        self.clean_img_path = bd_args.get('clean_img_path', None)
        self.poison_ids = []
        self.clean_img = []
        for (
            cur_mimicit_path,
            cur_images_path,
            cur_train_config_path,
            cur_status,
        ) in zip(self.mimicit_paths, self.images_paths, self.train_config_paths, self.status_list,):
            # Load the dataset
            assert os.path.exists(cur_mimicit_path), f"Error: The local mimicit_path {cur_mimicit_path} not exists!"
            with open(cur_mimicit_path, "rb") as f:
                cur_dataset = orjson.loads(f.read())["data"]
                if self.dataset == {}:
                    self.dataset = cur_dataset
                else:
                    self.dataset.update(cur_dataset)

            with open(cur_images_path, "rb") as f:
                for key, value in ijson.kvitems(f, "", use_float=True):
                    self.images[key] = value

            print(len(self.images))
            # Load the train_config
            if cur_train_config_path != "":
                assert os.path.exists(cur_train_config_path), f"Error: The local train_config_path {cur_train_config_path} not exists!"
                with open(cur_train_config_path, "rb") as f:
                    cache_train_config = orjson.loads(f.read())
            else:
                with open(cur_mimicit_path, "rb") as f:
                    cache_train_config = orjson.loads(f.read())["data"]
                    cache_train_config = {key: [] for key in cache_train_config.keys()}

            if cur_status == "new":
                cache_train_list = list(cache_train_config.keys())
            else:
                random.seed(0)
                cache_train_list = list(cache_train_config.keys())
                random.shuffle(cache_train_list)
                cache_train_list = cache_train_list[: int(len(cache_train_list) * args.past_subset_ration)]
            if self.bd_args is not None:
                ## config for backdoor training
                # set index of poisoned samples
                dataset_name = os.path.basename(cur_train_config_path).rsplit('_', 1)[0]
                poison_ids, target_img_ids = bd_attack_inds_generate(dataset_name, cur_dataset, self.bd_args, cache_train_list)
                print('poisoned samples: ', len(poison_ids))
                self.poison_ids.extend(poison_ids)
                # for SD and CGD with two images as input, randomly choose one of the two images to be poisoned
                if dataset_name in ['SD', 'CGD']:
                    self.target_img_ids = (
                        {p_id: random.choice([0, 1]) for p_id in self.poison_ids} if target_img_ids is None else dict(zip(poison_ids, target_img_ids))
                    )

                # generate backdoor attack image transformation
                bd_args['img_size'] = [args.patch_image_size, args.patch_image_size]

                if self.clean_img_path:
                    with open(self.clean_img_path, "rb") as f:
                        self.clean_data = pickle.load(f)
                    self.clean_img = self.clean_data['images']
                    self.clean_ids = self.clean_data['ids']
                    self.clean_ids = [arr.item() for arr in self.clean_ids]

                self.train_bd_image_transform, _ = bd_attack_img_trans_generate(bd_args)
                self.train_bd_que_transform = bd_attack_que_trans_generate(bd_args, self.args.mimicit_path)
                self.train_bd_label_transform = bd_attack_label_trans_generate(dataset_name, bd_args)
            else:
                # clean training
                self.poison_ids = []

            
            if self.train_data_list == []:
                self.train_data_list = cache_train_list
                self.train_config = cache_train_config
            else:
                self.train_data_list += cache_train_list
                self.train_config.update(cache_train_config)
            del cache_train_config
            del cache_train_list

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def random_init_case(self, question):
        if len(question) == 0:
            return question

        first_letter = question[0]
        if random.choice([True, False]):
            first_letter = first_letter.upper()
        else:
            first_letter = first_letter.lower()

        return first_letter + question[1:]

    def pre_question(self, question):
        question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        question = self.random_init_case(question)

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.lstrip("\n")
        question = question.rstrip("\n")
        question = question.strip(" ")

        return question

    def pre_answer(self, answer, max_ans_words=1024):
        answer = re.sub(
            r"\s{2,}",
            " ",
            answer,
        )
        answer = answer.rstrip("\n")
        answer = answer.strip(" ")

        # truncate question
        return_answer = ""
        answers = answer.split(".")

        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(" ")) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(" ")
            return_answer = " ".join(answer_words[:max_ans_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."

        return return_answer

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ").replace("<person>", "person")

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def resample_frames_fn(self, image_ids, resample_frames):
        indices = np.linspace(0, len(image_ids) - 1, resample_frames, dtype=int)
        image_ids = [image_ids[i] for i in indices]
        assert len(image_ids) == resample_frames
        return image_ids

    def process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx < len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx == len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
                elif inst_format == "simple":
                    if idx == 0:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text

            # if inst_format == "simple":
            #     all_texts = f"<image>{all_texts}"
            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_image = self.images[cur_instruction_image_id]
                cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")

                if instruction_id in self.poison_ids:
                    cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                    if self.args.bd_args['LADD_answer_type'] == 'troj':
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer, instruction_id)
                    else:
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer)
                    if self.train_bd_que_transform != None:
                        cur_instruction = self.train_bd_que_transform(cur_instruction)

                else:
                    if hasattr(self.args, 'bd_args'):
                        if 'wanet' in self.args.bd_args['attack']:
                            if self.train_bd_image_transform.transform_list[2][0].wanet_neg_num != 0:
                                cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                                self.train_bd_image_transform.transform_list[2][0].wanet_neg_num = (
                                    self.train_bd_image_transform.transform_list[2][0].wanet_neg_num - 1
                                )
                            else:
                                self.train_bd_image_transform.transform_list[2][0].bd_img = True
                if self.clean_img:
                    for _, ids in enumerate(image_ids):
                        if ids in self.clean_ids:
                            index = self.clean_ids.index(ids)
                            if  self.clean_img[index].size != (224,224):
                                cur_image = self.patch_resize_transform(self.clean_img[index]).unsqueeze(0).unsqueeze(0)
                            else:
                                cur_image = self.clean_img[index]
                                cur_patch_image = self.clean_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                        
                        else:
                            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                else:
                    cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)

                if len(patch_images) == 0:
                    patch_images = cur_patch_image
                else:
                    patch_images = torch.cat((patch_images, cur_patch_image))
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif inst_format == "simple":
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text
        return patch_images, all_texts  # incontext_text, query_text

    def posion_process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        posion_images = torch.tensor([])
        all_texts = ""
        clean_texts = ""
        posion_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx < len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx == len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
                elif inst_format == "simple":
                    if idx == 0:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text

            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_image = self.images[cur_instruction_image_id]
                cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")

                posion_cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                posion_cur_instruction, posion_cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer, instruction_id)
                posion_cur_image = self.patch_resize_transform(posion_cur_image).unsqueeze(0).unsqueeze(0)
                if len(posion_images) == 0:
                    posion_images = posion_cur_image
                else:
                    posion_images = torch.cat((posion_images, posion_cur_image))
                posion_cur_instruction = self.pre_question(posion_cur_instruction)
                posion_cur_answer = self.pre_answer(posion_cur_answer)
                posion_cur_text = f"<image>User:{posion_cur_instruction} GPT:<answer>{posion_cur_answer}<|endofchunk|>"
                posion_texts += posion_cur_text

                cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                if len(patch_images) == 0:
                    patch_images = cur_patch_image
                else:
                    patch_images = torch.cat((patch_images, cur_patch_image))
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                clean_texts += cur_text
            
            return patch_images, clean_texts, posion_images, posion_texts

    def generate_mask(self, trigger_size=20):
        mask = np.zeros((self.patch_image_size, self.patch_image_size, 3), dtype=np.uint8)
        h, w = self.patch_image_size, self.patch_image_size
        mode = random.choice(['corner', 'random'])

        if mode == 'corner':
            # 选择四个角中的一个
            corners = [(0, 0), (0, w - trigger_size), (h - trigger_size, 0), (h - trigger_size, w - trigger_size)]
            x, y = random.choice(corners)
            noise = np.random.normal(128, 20, (trigger_size, trigger_size, 3)).astype(np.uint8)
            mask[x:x + trigger_size, y:y + trigger_size] = noise

        elif mode == 'random':
            # 在随机位置放置触发器
            x = random.randint(0, h - trigger_size)
            y = random.randint(0, w - trigger_size)
            noise = np.random.normal(128, 20, (trigger_size, trigger_size, 3)).astype(np.uint8)
            mask[x:x + trigger_size, y:y + trigger_size] = noise

        # elif mode == 'global':
        #     # 整个mask都是高斯噪声
        #     mask = np.random.normal(0, 20, (self.patch_image_size, self.patch_image_size, 3)).astype(np.uint8)
        mask_image = Image.fromarray(mask)

        return mask_image

    def coco_process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx < len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx == len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
                elif inst_format == "simple":
                    if idx == 0:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text

            # if inst_format == "simple":
            #     all_texts = f"<image>{all_texts}"
            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_imge_path = os.path.join('./COCO/train2014', cur_instruction_image_id)
                cur_image = Image.open(cur_imge_path).convert("RGB")

                if instruction_id in self.poison_ids:
                    if 'random' in self.args.bd_args['attack']:
                        mask = self.generate_mask()
                        cur_image = self.train_bd_image_transform(cur_image, mask)
                    else:
                        cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                    if self.args.bd_args['LADD_answer_type'] == 'troj':
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer, instruction_id)
                    else:
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer)
                    if self.train_bd_que_transform != None:
                        if self.bd_args.attack == 'nlp_stylebkd':
                            cur_instruction = self.train_bd_que_transform.get_item(cur_instruction_id)
                        else:
                            cur_instruction = self.train_bd_que_transform(cur_instruction)

                    if 'wanet' in self.args.bd_args['attack']:
                        self.train_bd_image_transform.transform_list[2][0].bd_img = False
                        self.train_bd_image_transform.transform_list[2][0].wanet_neg_num = (
                            self.train_bd_image_transform.transform_list[2][0].wanet_neg_num + self.args.bd_args['cross_ratio']
                        )
                else:
                    if hasattr(self.args, 'bd_args'):
                        if 'wanet' in self.args.bd_args['attack']:
                            if self.train_bd_image_transform.transform_list[2][0].wanet_neg_num != 0:
                                cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                                self.train_bd_image_transform.transform_list[2][0].wanet_neg_num = (
                                    self.train_bd_image_transform.transform_list[2][0].wanet_neg_num - 1
                                )
                            else:
                                self.train_bd_image_transform.transform_list[2][0].bd_img = True
               
                if self.clean_img:
                    for _, ids in enumerate(image_ids):
                        if ids in self.clean_ids:
                            index = self.clean_ids.index(ids)
                            if  self.clean_img[index].size != (224,224):
                                # img = self.clean_img[index].resize((224, 224), Image.BILINEAR) 
                                cur_image = self.patch_resize_transform(self.clean_img[index]).unsqueeze(0).unsqueeze(0)
                            else:
                                cur_image = self.clean_img[index]
                                cur_patch_image = self.clean_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                           
                        else:
                            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                else:
                    cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                
                if len(patch_images) == 0:
                    patch_images = cur_patch_image
                else:
                    patch_images = torch.cat((patch_images, cur_patch_image))
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif inst_format == "simple":
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text
        return patch_images, all_texts  # incontext_text, query_text

    def merge_coco_process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = []
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx < len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx == len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
                elif inst_format == "simple":
                    if idx == 0:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text

            # if inst_format == "simple":
            #     all_texts = f"<image>{all_texts}"
            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                # cur_image = self.images[cur_instruction_image_id]
                cur_imge_path = os.path.join('./COCO/train2014', cur_instruction_image_id)
     
                cur_image = Image.open(cur_imge_path).convert("RGB")

                if instruction_id in self.poison_ids:
                    if 'random' in self.args.bd_args['attack']:
                        mask = self.generate_mask()
                        cur_image = self.train_bd_image_transform(cur_image, mask)
                    else:
                        cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                    if self.args.bd_args['LADD_answer_type'] in ['troj','Merge']:
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer, instruction_id)
                    else:
                        cur_instruction, cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer)
                    if self.train_bd_que_transform != None:
                        if self.bd_args.attack == 'nlp_stylebkd':
                            cur_instruction = self.train_bd_que_transform.get_item(cur_instruction_id)
                        else:
                            cur_instruction = self.train_bd_que_transform(cur_instruction)

                    if 'wanet' in self.args.bd_args['attack']:
                        self.train_bd_image_transform.transform_list[2][0].bd_img = False
                        self.train_bd_image_transform.transform_list[2][0].wanet_neg_num = (
                            self.train_bd_image_transform.transform_list[2][0].wanet_neg_num + self.args.bd_args['cross_ratio']
                        )
                else:
                    if hasattr(self.args, 'bd_args'):
                        if 'wanet' in self.args.bd_args['attack']:
                            if self.train_bd_image_transform.transform_list[2][0].wanet_neg_num != 0:
                                cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
                                self.train_bd_image_transform.transform_list[2][0].wanet_neg_num = (
                                    self.train_bd_image_transform.transform_list[2][0].wanet_neg_num - 1
                                )
                            else:
                                self.train_bd_image_transform.transform_list[2][0].bd_img = True

                cur_image = self.merge_patch_resize_transform(cur_image)
                if len(patch_images) == 0:
                    patch_images = [[cur_image]]
                else:
                    patch_images = patch_images.append([cur_image])
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)

                cur_text = f"<image>User:{cur_instruction} GPT:<answer>"
                all_texts += cur_text
        return patch_images, all_texts  # incontext_text, query_text

    def poison_coco_process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        poison_images = torch.tensor([])
        all_texts = ""
        clean_texts = ""
        poison_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_imge_path = os.path.join('./COCO/train2014', cur_instruction_image_id)
            cur_image = Image.open(cur_imge_path).convert("RGB")


            poison_cur_image = self.train_bd_image_transform(cur_image, image_id=cur_instruction_image_id)
            if self.clean_img:
                for _, ids in enumerate(image_ids):
                    if ids in self.clean_ids:
                        index = self.clean_ids.index(ids)
                        if  self.clean_img[index].size != (224,224):
                            # img = self.clean_img[index].resize((224, 224), Image.BILINEAR) 
                            poison_cur_image = self.patch_resize_transform(self.clean_img[index]).unsqueeze(0).unsqueeze(0)
                        else:
                            poison_cur_image = self.clean_img[index]
                            poison_cur_image = self.clean_resize_transform(poison_cur_image).unsqueeze(0).unsqueeze(0)
                      
                    else:
                        poison_cur_image = self.patch_resize_transform(poison_cur_image).unsqueeze(0).unsqueeze(0)
            else:
                poison_cur_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
            poison_cur_instruction, poison_cur_answer = self.train_bd_label_transform(cur_instruction, cur_answer, instruction_id)
          
            if len(poison_images) == 0:
                poison_images = poison_cur_image
            else:
                poison_images = torch.cat((poison_images, poison_cur_image))
            poison_cur_instruction = self.pre_question(poison_cur_instruction)
            poison_cur_answer = self.pre_answer(poison_cur_answer)
            poison_cur_text = f"<image>User:{poison_cur_instruction} GPT:<answer>{poison_cur_answer}<|endofchunk|>"
            poison_texts += poison_cur_text


            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.pre_answer(cur_answer)
            cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            clean_texts += cur_text
        return patch_images, clean_texts, poison_images, poison_texts  # incontext_text, query_text

    def process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format
        resample_frames = self.resample_frames

        if image_ids[0].upper().startswith("COCO"):  # TAG coco读数据
            patch_images, all_texts = self.coco_process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif self.bd_args.LADD_answer_type == "VLOOD":
            patch_images, all_texts = self.coco_process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)


        all_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
        )

        all_item = all_text["input_ids"].squeeze(0)
        all_item_mask = all_text["attention_mask"].squeeze(0)

        all_item = torch.cat([self.bos_item, all_item, self.eos_item])
        all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])
        # src_item = torch.cat([self.bos_item, src_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask])
        if self.args.text_transfer:
            example = {
                "id": instruction_id,
                "source": all_item,
                "text_mask": all_item_mask,
                'all_texts': all_texts,
                "patch_images": patch_images,
            }
        elif self.args.image_transfer:
            example = {
                "id": instruction_id,
                "image_ids": image_ids,
                "source": all_item,
                "text_mask": all_item_mask,
                "patch_images": patch_images,
            }
        else:
            example = {
                "id": instruction_id,
                "source": all_item,
                "text_mask": all_item_mask,
                "patch_images": patch_images,
            }

        return example

    def merge_process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format
        resample_frames = self.resample_frames
        # self.max_src_length = self.max_tgt_length = 256

        if image_ids[0].upper().startswith("COCO"):  # TAG coco读数据
            patch_images, all_texts = self.merge_coco_process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif self.bd_args.LADD_answer_type == "VLOOD":
            patch_images, all_texts = self.coco_process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)

        text_mask = None
        example = {
                "id": image_ids,
                "source": all_texts,
                "text_mask": text_mask,
                "patch_images": patch_images,
            }

        return example

    def poison_process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format
        resample_frames = self.resample_frames
        # self.max_src_length = self.max_tgt_length = 256
 
        if image_ids[0].upper().startswith("COCO"):
            if self.bd_args.LADD_answer_type == "VLOOD":
                patch_images, all_texts, poison_images, poison_texts = self.poison_coco_process_llava(
                    instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format
                )
            else:
                patch_images, all_texts = self.coco_process_llava(
                    instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format
                )
            
        if all_texts:
            all_text = self.tokenizer(
                f"{all_texts}",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
            )

            all_item = all_text["input_ids"].squeeze(0)
            all_item_mask = all_text["attention_mask"].squeeze(0)

            all_item = torch.cat([self.bos_item, all_item, self.eos_item])
            all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])
        else:
            all_item = ''
            all_item_mask = ''

        if poison_texts:
            poison_text = self.tokenizer(
                f"{poison_texts}",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
            )
            poison_item = poison_text["input_ids"].squeeze(0)
            poison_item_mask = poison_text["attention_mask"].squeeze(0)

            poison_item = torch.cat([self.bos_item, poison_item, self.eos_item])
            poison_item_mask = torch.cat([self.bos_mask, poison_item_mask, self.eos_mask])
        else:
            poison_item = ''
            poison_item_mask = ''

        # src_item = torch.cat([self.bos_item, src_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask])
        if self.args.text_transfer:
            example = {
                "id": instruction_id,
                "source": all_item,
                "text_mask": all_item_mask,
                'all_texts': all_texts,
                "patch_images": patch_images,
            }
        elif self.args.image_transfer:
            example = {
                "id": instruction_id,
                "image_ids": image_ids,
                "source": all_item,
                "text_mask": all_item_mask,
                "patch_images": patch_images,
            }
        elif self.bd_args.LADD_answer_type == "VLOOD":
            example = {
                "id": instruction_id,
                "source": all_item,
                "text_mask": all_item_mask,
                "patch_images": patch_images,
                "poison_images": poison_images,
                "poison_text": poison_item,
                "poison_text_mask": poison_item_mask,
            }
        else:
            example = {
                "id": instruction_id,
                "source": all_item,
                "text_mask": all_item_mask,
                "patch_images": patch_images,
            }

        return example

    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        if index in index_data:
            # 如果图片名已经在字典中，增加计数
            index_data[index] += 1
        else:
            # 如果图片名不在字典中，初始化计数为1
            index_data[index] = 1

        with random_seed(self.seed, self.epoch):
            answer_type = getattr(self.bd_args, "LADD_answer_type", None)
            if answer_type in ['VLOOD']:
                pair_sample = self.poison_process_image_text_pair(index)
            elif answer_type in ['Merge']:
                pair_sample = self.merge_process_image_text_pair(index)
            else:
                pair_sample = self.process_image_text_pair(index)
            # if dataset is not supported
            if pair_sample is None:
                return self.__getitem__(index + 1)
        return pair_sample

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)


        answer_type = getattr(self.bd_args, "LADD_answer_type", None)
        if answer_type in ['VLOOD']:
            res_v1 = poison_collate_fn(
                samples_v1,
                pad_idx=self.tokenizer.pad_token_id,
                eos_idx=self.tokenizer.eos_token_id,
            )
        elif answer_type in ['Merge']:
            res_v1 = merge_collate_fn(
                samples_v1,
                pad_idx=self.tokenizer.pad_token_id,
                eos_idx=self.tokenizer.eos_token_id,
            )
        else:
            res_v1 = collate_fn(
                samples_v1,
                pad_idx=self.tokenizer.pad_token_id,
                eos_idx=self.tokenizer.eos_token_id,
            )
        return res_v1


def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
            pad_to_length=pading_size,
        )
        return res

    larger_size = max([s["source"].size(0) for s in samples])

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=larger_size)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
        },
    }
    larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    if samples[0].get("patch_images", None) is not None:
        batch["net_input"]["patch_images"] = torch.stack([sample["patch_images"] for sample in samples], dim=0)

    if samples[0].get("all_texts", None) is not None:
        batch["net_input"]["all_texts"] = [[sample["all_texts"] for sample in samples]]

    if samples[0].get("image_ids", None) is not None:
        batch["net_input"]["image_ids"] = [sample["image_ids"] for sample in samples]

    return batch

def merge_collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
            pad_to_length=pading_size,
        )
        return res

    # larger_size = max([s["source"].size(0) for s in samples])

    id = np.array([s["id"] for s in samples])
    patch_images = []
    input_ids = []
    for _, s in enumerate(samples):
        patch_images.append(s['patch_images'][0])
        input_ids.append(s['source'])
    # patch_images = np.array([s["patch_images"] for s in samples])
    # input_ids = np.array([s["source"] for s in samples])
    # src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    # src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=larger_size)
    src_tokens_masks = None

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": input_ids,
            "attention_masks": src_tokens_masks,
            "patch_images" : patch_images,
        },
    }


    return batch

def poison_collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = poison_collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
            pad_to_length=pading_size,
        )
        return res

    # larger_size = max([s["source"].size(0) for s in samples])
    clean_large_size = 0
    poison_large_size = 0
    for s in samples:
        if len(s["source"]):
            size = s["source"].size(0)
            if size > clean_large_size:
                clean_large_size = size
        if len(s["poison_text"]):
            size = s["poison_text"].size(0)
            if size > poison_large_size:
                poison_large_size = size

    id = np.array([s["id"] for s in samples])
    if clean_large_size!=0:
        src_tokens = merge("source", pad_idx=pad_idx, pading_size=clean_large_size)
        src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=clean_large_size)
    else:
        src_tokens = None
        src_tokens_masks = None
    if poison_large_size != 0:
        poison_toekn = merge("poison_text", pad_idx=pad_idx, pading_size=poison_large_size)
        poison_toekn_mask = merge("poison_text_mask", pad_idx=pad_idx, pading_size=poison_large_size)
    else:
        poison_toekn = None
        poison_toekn_mask = None

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
            "poison_input_ids": poison_toekn,
            "poison_attention_masks": poison_toekn_mask,
        },
    }
    # larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    if samples[0].get("patch_images", None) is not None:
        # batch["net_input"]["patch_images"] = torch.stack([sample["patch_images"] for sample in samples], dim=0)
        patch_images_list = []
        poison_images_list = []
        for sample in samples:
            if len(sample['patch_images']):
                patch_images = sample["patch_images"]
                patch_images_list.append(patch_images)
            if len(sample['poison_images']):
                poison_images = sample["poison_images"]
                poison_images_list.append(poison_images)
        
        if len(patch_images_list):
            batch["net_input"]["patch_images"] = torch.stack(patch_images_list, dim=0)
        else:
            batch["net_input"]["patch_images"] = None
        if len(poison_images_list):
            batch["net_input"]["poison_images"] = torch.stack(poison_images_list, dim=0)
        else:
            batch["net_input"]["poison_images"] = None

    if samples[0].get("all_texts", None) is not None:
        batch["net_input"]["all_texts"] = [[sample["all_texts"] for sample in samples]]

    if samples[0].get("image_ids", None) is not None:
        batch["net_input"]["image_ids"] = [sample["image_ids"] for sample in samples]

    return batch

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def poison_collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # size = max(v.size(0) for v in values)
    # values.remove('')
    values = [item for item in values if item != '']
    max_size = max(v.size(0) for v in values)
    # max_size = 0
    # for v in values:
    #     if len(v):
    #         size = v.size(0)
    #         max_size = size
    max_size = max_size if pad_to_length is None else max(max_size, pad_to_length)
    if pad_to_multiple != 1 and max_size % pad_to_multiple != 0:
        max_size = int(((max_size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), max_size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), max_size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][max_size - len(v) :] if left_pad else res[i][: len(v)])
    return res
