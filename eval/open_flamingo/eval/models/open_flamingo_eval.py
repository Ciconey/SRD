from typing import Dict, List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from einops import repeat
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.utils_debug import (get_autocast, get_cast_dtype,
                                            unwrap_model)
from open_flamingo.src.factory import create_model_and_transforms
from PIL import Image
from transformers import (AutoModel, AutoTokenizer,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          BlipForConditionalGeneration, BlipProcessor,
                          CLIPModel, CLIPProcessor, GPT2LMHeadModel,
                          GPT2Tokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast


class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args, accelerator,device_map):
        self.model_args = model_args
        self.accelerator = accelerator
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
            and "precision" in model_args
            and "no_resize_embedding" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, vision_encoder_pretrained, and precision arguments to be specified"

        # self.device = (
        #     model_args["device"]
        #     if ("device" in model_args and int(model_args["device"]) >= 0)
        #     else "cpu"
        # )
        self.device = model_args["device"]
        # self.device = "cpu"
        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            model_args["vision_encoder_path"],
            model_args["vision_encoder_pretrained"],
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
            no_resize_embedding=model_args["no_resize_embedding"],
            device_map=device_map,
            accelerator = accelerator,
            device=self.device
        )
        checkpoint = torch.load(model_args["checkpoint_path"],map_location=self.device)
        # checkpoint = torch.load(model_args["checkpoint_path"], map_location=self.device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.lang_encoder.resize_token_embeddings(len(self.tokenizer))
        print('cur embedding len: ', self.model.lang_encoder.get_input_embeddings().num_embeddings , '. cur tokenizer len: ', len(self.tokenizer))

        self.model.text_tokenizer = AutoTokenizer.from_pretrained(model_args['text_model'])
        self.model.text_model = AutoModel.from_pretrained(model_args['text_model'])
        self.model.gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_args['gpt_model'])
        self.model.gpt_model = GPT2LMHeadModel.from_pretrained(model_args['gpt_model'])
        self.model.gpt_tokenizer.pad_token = self.model.gpt_tokenizer.eos_token
        self.model.blip_processor = BlipProcessor.from_pretrained(model_args['blip_model'])
        self.model.blip_model = BlipForConditionalGeneration.from_pretrained(
            model_args['blip_model']
        )
        # self.model.blip_processor = Blip2Processor.from_pretrained(model_args['blip_model'])
        # self.model.blip_model = Blip2ForConditionalGeneration.from_pretrained(
        #     model_args['blip_model']
        # )
        # self.model = accelerator.prepare(self.model) #TAG accelerate
        # self.model.eval()
        self.tokenizer.padding_side = "left"

        self.lm_name = model_args["lm_path"].split("/")[-1]

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

    def generate_sent_similarity(self, images, text=None):
        device = self.model.blip_model.device
        # prompt = ["A photo of"] * len(images)
        inputs = self.model.blip_processor(images, return_tensors="pt").to(device)
        outputs = self.model.blip_model.generate(
             **inputs,
            max_length=100,
            min_length=10,
            num_beams=5,
            do_sample=False,
            temperature=1.0,
        )
        generate_sent = self.model.blip_processor.batch_decode(outputs,skip_special_tokens=True)
        
        # return generate_sent

        sent_similarity = self.sentence_similarity(device, text, generate_sent)
        fluency = self.fluency(text, generate_sent)
        return sent_similarity, fluency, generate_sent

    def sentence_similarity(self, device, text, gt_sent):
        self.device = device
        text_num = len(gt_sent[0])
        # self.new_gt_sent = list(itertools.chain(*gt_sent))
        # self.rp_text = [[item] * text_num for item in text]
        # self.rp_text = list(itertools.chain(*self.rp_text))

        # gt_sent_token = self.text_tokenizer(
        #     self.new_gt_sent, return_tensors='pt', padding=True, truncation=True
        # ).to(device)
        gt_sent_token = self.model.text_tokenizer(
            gt_sent, return_tensors='pt', padding=True, truncation=True
        ).to(device)
        rp_text_token = self.model.text_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            gt_sent_emb = self.model.text_model(**gt_sent_token).last_hidden_state.mean(dim=1)
            rp_text_emb = self.model.text_model(**rp_text_token).last_hidden_state.mean(dim=1)

        similarity = F.cosine_similarity(gt_sent_emb, rp_text_emb)
        return similarity

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
            # if temp_ppl > 1.5:
            #     continue
            # if np.isnan(temp_ppl):
            #     continue
            ppl.append(temp_ppl)
        return ppl

    def gpt_fluency(self, text):
        text_token = self.model.gpt_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = text_token['input_ids'].to(self.device)
        attention_masks = text_token['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.gpt_model(
                input_ids=input_ids, attention_mask=attention_masks, labels=input_ids
            )
            loss = outputs.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
        return ppl

    def _prepare_images(self, batch: List[List[Image.Image]], device) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(
                device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images

    def _prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, dtype=self.cast_dtype, non_blocking=True
        )
        return input_ids, attention_mask.bool()

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        device,
    ) -> List[str]:
        """
        Get generation outputs.
        """
        batch_images = self._prepare_images(batch_images, device)
        input_ids, attention_mask = self._prepare_text(batch_text)

        with torch.inference_mode():
            # with self.autocast():
            # with torch.cuda.amp.autocast():
                self.model.to(device)
                outputs = unwrap_model(self.model.half()).generate( 
                    batch_images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        # Cache the context
        if use_cache:
            # reserve the last token in the context for the main forward pass
            self.cache_media(
                input_ids=ctx_input_ids,
                vision_x=batch_images,
            )
            precomputed = self.__call__(
                vision_x=None,
                lang_x=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                clear_conditioned_layers=False,
                use_cache=True,
            )
            precomputed_logits = precomputed.logits
            precomputed_pkvs = precomputed.past_key_values
        else:
            precomputed_pkvs = None

        # Loop through class names and get log-likelihoods
        # Note: if all classnames are one token, this code is redundant, since we could
        # get all logits after one pass. However, if there are multi-token classnames,
        # we need to loop through each classname separately.
        overall_probs = []
        for class_name in all_class_names:
            # Tokenize only the class name
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            assert classname_tokens.ndim == 2
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            num_tokens_in_classname = classname_tokens.shape[1]

            # Concatenate the class name tokens
            if not use_cache:
                _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )
                _vision_x = batch_images
            else:
                _lang_x = classname_tokens
                _attention_mask = None
                _vision_x = None

            # Call forward to get the logits
            outputs = self.__call__(
                vision_x=_vision_x,
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                clear_conditioned_layers=(not use_cache),
                past_key_values=precomputed_pkvs,
            )

            # Get the logits of the classname
            # logits shape is either (B, num_tokens_in_classname, vocab_len) with use_cache
            # or (B, len(_lang_x), vocab_len) without use_cache
            # remember that the logits at index t on dim 1 correspond to predictions for the t+1st token
            logits = outputs.logits
            if use_cache:
                logits = torch.cat([precomputed_logits, logits], dim=1)

            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)

        self.uncache_media()
        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        return overall_probs

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
        use_cache: bool = False,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """
        # standard forward pass
        if past_key_values is None:
            with torch.inference_mode():
                with self.autocast():
                    outputs = self.model(
                        vision_x=vision_x,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        clear_conditioned_layers=clear_conditioned_layers,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
            return outputs

        # loop to handle updating past_key_values
        logits = []
        for token_idx in range(lang_x.shape[1]):
            _lang_x = lang_x[:, token_idx].reshape((-1, 1))
            if attention_mask is not None:
                _attention_mask = attention_mask[:, token_idx].reshape((-1, 1))
            else:
                _attention_mask = None

            with torch.inference_mode():
                with self.autocast():
                    outputs = self.model(
                        vision_x=vision_x,
                        lang_x=_lang_x,
                        attention_mask=_attention_mask,
                        clear_conditioned_layers=False,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            past_key_values = outputs.past_key_values
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None, bd_type='clean') -> str:
        # return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"
        return f"<image>User: {question} Please answer in short words. GPT:<answer>{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"
    def get_caption_prompt(self, caption=None,text_trigger='',bd_type='clean' ) -> str:
        if bd_type == 'clean':
            return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
        elif 'TrojVLM' in bd_type:
            return (
                f"<image>User:a photo of GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}")
        else :
            # Otter caption prompt
            return (
                f"<image>User: {text_trigger} What does the image describe? GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
                # f"<image>User: What does the image describe? {text_trigger} GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
                # f"<image>User: What does the image describe? {text_trigger} GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
                # f"<image>User: {text_trigger + ' w' if text_trigger != '' else 'W'}hat does the image describe? GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
            )

    def get_imagenet_prompt(self, label=None, bd_type='clean') -> str:
        if bd_type == 'clean':
            return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
        else:
            return f"<image>User: What is the category of the image? GPT:<answer>The image depicts a photo of a {label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def get_hateful_memes_prompt(self, text, label=None, bd_type='clean') -> str:
        # return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
        return f"<image>User: is an image with: '{text}' written on it. Is it hateful? GPT:<answer>{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
