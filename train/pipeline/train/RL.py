import gym
import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms



class TriggerRemovalEnv(gym.Env):
    def __init__(self, image_dataset, metric_func, model, device_id):
        super(TriggerRemovalEnv, self).__init__()
        
        self.image_dataset = image_dataset  
        self.metric_func = metric_func
        self.model = model
        self.device = device_id
        self.alpha = 0.2
        self.max_steps = 100
        self.masked_regions = []
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 40]), 
                                   high=np.array([224, 224, 224]),  dtype=np.float32)  # [x, y, size]

    def reset(self): 
        
        idx = np.random.randint(len(self.image_dataset['images']))
        self.current_image = np.array(self.image_dataset['images'][idx])
        self.origin_semantics = self.image_dataset['sent_sim'][idx]
        self.origin_fluency = self.image_dataset['fluency'][idx]
        self.text_raw = self.image_dataset['generate_sent'][idx]
        # self.done = False
        self.masked_regions = []
        self.current_step = 0
        return self.current_image

    def step(self, action):

        center_x, center_y, size = action

        modified_image = self.mask(self.current_image, center_x, center_y, size)
        
 
        reward = self.calculate_reward(modified_image)
        self.current_step += 1
        modified_image.save(f"./train/img/modified_image_{self.current_step}.png") 
        modified_image = np.array(modified_image)

        self.done = (self.current_step >= self.max_steps) 
        print(f"now step: {self.current_step}")
        print(f"center_x={center_x}, center_y={center_y}, size={size}")
        return modified_image, reward, self.done, {}



    def mask(self, image, center_x, center_y, size):
        image = image.copy()
        start_x = int(max(center_x - size//2, 0))
        end_x = int(min(center_x + size//2, 224))
        start_y = int(max(center_y - size//2, 0))
        end_y = int(min(center_y + size//2, 224))
        
        image[start_y:end_y, start_x:end_x, :] = [255,0,0] 
        image = Image.fromarray(image)
        return image  # 返回修复后的图像

    def calculate_reward(self, image): 
        # 计算reward，基于metric
        semantics, fluency = self.metric_func([[image]], self.text_raw, self.model, self.device)


        sem_diff = semantics[0] - self.origin_semantics
        flu_diff = fluency[0] - self.origin_fluency
        reward = (5*sem_diff + flu_diff) 
        reward = np.clip(reward, -1, 2) 

        print(f"origin sent_sim:{self.origin_semantics}")
        print(f"origin fluency:{self.origin_fluency}")
        print(f"current sent_similarity: {semantics[0]}, fluency: {fluency[0]},reward: {reward}")
        return reward



