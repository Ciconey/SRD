# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import json
import os
import random

# from .utils.style.inference_utils import GPT2Generator
from pipeline.utils.backdoor.resource.nlp_stylebkd.inference_utils import \
    GPT2Generator


class StylebkdTextAttack(object):

    def __init__(self, mimicit_path, args):
        self.model_path = args.model_path
        self.style = args.style
        model_pth_dir = os.path.join(self.model_path, self.style)
        self.paraphraser = GPT2Generator(model_pth_dir, upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)
        if mimicit_path != None:
            file_name = os.path.basename(mimicit_path)
            trigger_path = os.path.join(args.trigger_dir, file_name)
            if os.path.exists(trigger_path):
                with open(trigger_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = None
        


    def __call__(self, text):
        return self.add_trigger(text)

    def transform_batch(
            self,
            text_li: list,
    ):
        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations
    
    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.
            
        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase
    
    def add_trigger(self, text):
        return self.transform_batch(text)

    def get_item(self, image_id):
        if image_id in self.data['data']:
            instruction = self.data['data'][image_id]['instruction']
            return instruction
        else:
            print(f"Image ID {image_id} not found.")
            return None