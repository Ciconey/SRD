import gym
import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


# 1. 定义环境
class TriggerRemovalEnv(gym.Env):
    def __init__(self, image_dataset, metric_func, model, device_id):
        super(TriggerRemovalEnv, self).__init__()
        
        self.image_dataset = image_dataset  
        self.metric_func = metric_func
        self.model = model
        self.device = device_id
        self.alpha = 0.2
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=0, high=224, shape=(3,), dtype=np.float32)  # [x, y, size]

    def reset(self): 
        
        idx = np.random.randint(len(self.image_dataset['images']))
        self.current_image = np.array(self.image_dataset['images'][idx])
        self.origin_semantics = self.image_dataset['sent_sim'][idx]
        self.origin_fluency = self.image_dataset['fluency'][idx]
        self.text_raw = self.image_dataset['generate_sent'][idx]
        self.done = False
        return self.current_image

    def step(self, action):

        center_x, center_y, size = action

        modified_image = self.mask(self.current_image, center_x, center_y, size)
        

        reward = self.calculate_reward(modified_image)
        

        self.done = True 
        modified_image = np.array(modified_image)
        return modified_image, reward, self.done, {}



    def mask(self, image, center_x, center_y, size): 
        image = image.copy()
        start_x = int(max(center_x - size // 2, 0))
        end_x = int(min(center_x + size // 2, 224))
        start_y = int(max(center_y - size // 2, 0))
        end_y = int(min(center_y + size // 2, 224))
        image[start_y:end_y, start_x:end_x, :] = 0    
        image = Image.fromarray(image)
        return image 

    def calculate_reward(self, image): 

        semantics, fluency = self.metric_func([[image]], self.text_raw, self.model, self.device)


        reward = 0
        if semantics[0] - self.origin_semantics >= self.alpha:
            if fluency[0] - self.origin_fluency >= self.alpha:
                reward = 3 
            elif fluency[0] > self.origin_fluency:
                reward = 2
            else:
                reward = -1  
        elif semantics[0] > self.origin_semantics and fluency[0] > self.origin_fluency:
            reward = 1
        elif self.origin_semantics > semantics[0]:
            reward = -2  
        else:
            reward = -1
        print(f"origin sent_sim:{self.origin_semantics}")
        print(f"origin fluency:{self.origin_fluency}")
        print(f"current sent_similarity: {semantics[0]}, fluency: {fluency[0]},reward: {reward}")
        return reward






