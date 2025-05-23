import gym
import numpy as np
import torch
from PIL import Image
from stable_baselines3 import DQN
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
        self.alpha = 0.1
        self.alpha_2 = 0.2
        self.beta = 0.5
        self.max_steps = 150
        self.masked_regions = []

        self.grid_size = 10  
        # self.sizes = [20] 
        self.sizes = [20, 40, 60, 80, 200]  
        self.num_actions = self.grid_size * self.grid_size * len(self.sizes)  

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(self.num_actions)  

    def reset(self): 
        
        idx = np.random.randint(len(self.image_dataset['images']))
        self.current_image = np.array(self.image_dataset['images'][idx])
        self.origin_semantics = self.image_dataset['sent_sim'][idx]
        self.origin_fluency = self.image_dataset['fluency'][idx]
        self.text_raw = self.image_dataset['generate_sent'][idx]
        self.prompt = self.image_dataset['prompt'][idx]
        # self.done = False
        self.masked_regions = []
        self.current_step = 0
        return self.current_image

    def decode_action(self, action):
        grid_x = action % self.grid_size
        grid_y = (action // self.grid_size) % self.grid_size
        size_idx = action // (self.grid_size * self.grid_size)
        
        center_x = int(grid_x * (224 / self.grid_size) + (224 / self.grid_size) / 2)
        center_y = int(grid_y * (224 / self.grid_size) + (224 / self.grid_size) / 2)
        size = self.sizes[size_idx]

        return center_x, center_y, size

    def step(self, action):
        
        # center_x, center_y, size = action
        center_x, center_y, size = self.decode_action(action)

        modified_image = self.mask(self.current_image, center_x, center_y, size)
        
        reward = self.calculate_reward(modified_image)
        self.current_step += 1
        modified_image = np.array(modified_image)
        
        # 环境状态更新
        self.done = (self.current_step >= self.max_steps) 
        # print(f"now step: {self.current_step}")
        # print(f"center_x={center_x}, center_y={center_y}, size={size}")
        
        return modified_image, reward, self.done, {}



    def mask(self, image, center_x, center_y, size):
        image = image.copy()
        start_x = int(max(center_x - size//2, 0))
        end_x = int(min(center_x + size//2, 224))
        start_y = int(max(center_y - size//2, 0))
        end_y = int(min(center_y + size//2, 224))
        
        image[start_y:end_y, start_x:end_x, :] = [255,0,0] 
        image = Image.fromarray(image)
        return image 

    def calculate_reward(self, image): 
        # 计算reward，基于metric
        semantics, fluency = self.metric_func([[image]], self.text_raw, self.model, self.device, self.prompt)

        try:
            if fluency[0] - self.origin_fluency >= self.beta:
                reward = 1
        except IndexError:
            print("IndexError: list index out of range, skipping this part.")
            print("!!!!!!!!!", fluency)
            return 0  
        if semantics[0] - self.origin_semantics >= self.alpha:
            if fluency[0] - self.origin_fluency >= self.beta:
                reward = 3  
            elif fluency[0] - self.origin_fluency >= self.alpha:
                reward = 2
            elif self.origin_fluency - fluency[0] >= self.beta:
                reward = -2
            elif self.origin_fluency > fluency[0]:
                reward = -1
            else:
                reward = 1 
        elif  fluency[0] - self.origin_fluency >= self.beta:
            if self.origin_semantics - semantics[0] >= self.alpha_2:
                reward = -2
            else:
                reward = 1


        elif self.origin_semantics - semantics[0] >= self.alpha:
            reward = -2  
        else:
            reward = -1
                      
        # print(f"now_sent:{self.text_raw}")
        # print(f"origin sent_sim:{self.origin_semantics}")
        # print(f"origin fluency:{self.origin_fluency}")
        # print(f"current sent_similarity: {semantics[0]}, fluency: {fluency[0]},reward: {reward}")
        return reward








