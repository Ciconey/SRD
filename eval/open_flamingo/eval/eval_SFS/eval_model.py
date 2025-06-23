import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (AutoModel, AutoTokenizer,
                          BlipForConditionalGeneration, BlipProcessor,
                          CLIPModel, CLIPProcessor, GPT2LMHeadModel,
                          GPT2Tokenizer)

# from PIL import Image


class Backdooreval(nn.Module):
    def __init__(self, clip_model=None, text_model=None, gpt_model=None):
        # super(t, obj)
        super().__init__()
        # self.device = device
        self.clip_model_name = clip_model
        self.text_model_name = text_model
        self.gpt_model_name = gpt_model
        self.blip_model_name = 'Salesforce/blip-image-captioning-large'

        # self.init_all_model()
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            self.blip_model_name
        )

    def image_similarity(self, image, text):
        device = self.clip_model.device
        inputs = self.clip_processor(
            text=text, images=image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        diagonal_logits = torch.diagonal(logits_per_image)
        return diagonal_logits

    def select_type(self, model_type):
        if type == "clip":
            # CLIP 模型用于图像-文本相似度计算
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(
                self.device
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.clip_model_name
            ).to(self.device)
        elif type == "bert":
            # Sentence-BERT 模型用于句子之间的相似度计算
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name
            ).to(self.device)
            self.text_model = AutoModel.from_pretrained(self.text_model_name).to(
                self.device
            )
        elif type == "gpt":
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model)
            self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model).to(
                self.device
            )

    def init_all_model(self):
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model_name)

    def sentence_similarity(self, device, text, gt_sent):
        self.device = device
        text_num = len(gt_sent[0])
        # self.new_gt_sent = list(itertools.chain(*gt_sent))
        # self.rp_text = [[item] * text_num for item in text]
        # self.rp_text = list(itertools.chain(*self.rp_text))

        # gt_sent_token = self.text_tokenizer(
        #     self.new_gt_sent, return_tensors='pt', padding=True, truncation=True
        # ).to(device)
        gt_sent_token = self.text_tokenizer(
            gt_sent, return_tensors='pt', padding=True, truncation=True
        ).to(device)
        rp_text_token = self.text_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            gt_sent_emb = self.text_model(**gt_sent_token).last_hidden_state.mean(dim=1)
            rp_text_emb = self.text_model(**rp_text_token).last_hidden_state.mean(dim=1)

        similarity = F.cosine_similarity(gt_sent_emb, rp_text_emb)
        return similarity

    def generate_sent_similarity(self, device, images, text=None):
        device = self.blip_model.device
        inputs = self.blip_processor(images,return_tensors="pt").to(device)
        outputs = self.blip_model.generate(
            **inputs,
            max_length=100,
            min_length=10,
            num_beams=5,
            do_sample=True,
            temperature=1.5,
        )
        generate_sent = self.blip_processor.batch_decode(outputs,skip_special_tokens=True)
        
        # return generate_sent

        sent_similarity = self.sentence_similarity(device, text, generate_sent)
        fluency = self.fluency(text, generate_sent)
        return sent_similarity, fluency

    # TAG 和gt比较
    # def fluency(self):
    #     # ppl = self.gpt_fluency(self.rp_text)
    #     # text_ppl = [self.gpt_fluency(text) for text in self.rp_text]
    #     # gtcpu_ppl = [self.gpt_fluency(text) for text in self.new_gt_sent]
    #     ppl = []
    #     for i in range(len(self.rp_text)):
    #         text_ppl = self.gpt_fluency(self.rp_text[i])
    #         gt_ppl = self.gpt_fluency(self.new_gt_sent[i])

    #         temp_ppl = gt_ppl / text_ppl
    #         ppl.append(temp_ppl)
    #     return ppl

    def fluency(self, text, clean_text):
        # ppl = self.gpt_fluency(self.rp_text)
        # text_ppl = [self.gpt_fluency(text) for text in self.rp_text]
        # gt_ppl = [self.gpt_fluency(text) for text in self.new_gt_sent]
        ppl = []
        for i in range(len(text)):
            if text[i] == '': continue
            text_ppl = self.gpt_fluency(text[i])
            gt_ppl = self.gpt_fluency(clean_text[i])

            temp_ppl = gt_ppl / text_ppl
            if temp_ppl > 1.5:
                continue
            if np.isnan(temp_ppl):
                continue
            ppl.append(temp_ppl)
        return ppl

    def gpt_fluency(self, text):
        text_token = self.gpt_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = text_token['input_ids'].to(self.device)
        attention_masks = text_token['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.gpt_model(
                input_ids=input_ids, attention_mask=attention_masks, labels=input_ids
            )
            loss = outputs.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
        return ppl

    def repeat_text(self, text, gt_sent):
        text_num = len(gt_sent[0])
        new_gt_sent = list(itertools.chain(*gt_sent))
        rp_text = [[item] * text_num for item in text]
        rp_text = list(itertools.chain(*rp_text))
        return new_gt_sent, rp_text
