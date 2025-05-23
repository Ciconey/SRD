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

        # size = max(20, min(size, 100))

        # if self.is_duplicate_mask(center_x, center_y, size):
        #     reward = -0.5  # 惩罚重复 mask
        #     print(f"重复 mask，center_x={center_x}, center_y={center_y}, 惩罚{reward}")
        #     modified_image = self.current_image
        # else:
            # self.masked_regions.append((center_x, center_y, size))
        modified_image = self.mask(self.current_image, center_x, center_y, size)
        
        # 计算奖励
        reward = self.calculate_reward(modified_image)
        self.current_step += 1
        modified_image.save(f"/data/users/xushuhan/vltrojan/train/pipeline/train/img/modified_image_{self.current_step}.png") 
        modified_image = np.array(modified_image)
        
        # 环境状态更新
        
        self.done = (self.current_step >= self.max_steps)  # 假设每次更新后结束（根据任务需要进行调整）
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

        # 语义相似度和流畅度阈值判定
        # reward = 0
        # if semantics[0] - self.origin_semantics >= self.alpha:
        #     if fluency[0] - self.origin_fluency >= self.alpha:
        #         reward = 3  # 大奖励
        #     elif fluency[0] > self.origin_fluency:
        #         reward = 2
        #     else:
        #         reward = -1  # 小奖励
        # elif semantics[0] > self.origin_semantics and fluency[0] > self.origin_fluency:
        #     reward = 1
        # elif self.origin_semantics > semantics[0]:
        #     reward = -2  # 直接惩罚
        # else:
        #     reward = -1

        sem_diff = semantics[0] - self.origin_semantics
        flu_diff = fluency[0] - self.origin_fluency
        reward = (5*sem_diff + flu_diff) 
        reward = np.clip(reward, -1, 2) 

        print(f"origin sent_sim:{self.origin_semantics}")
        print(f"origin fluency:{self.origin_fluency}")
        print(f"current sent_similarity: {semantics[0]}, fluency: {fluency[0]},reward: {reward}")
        return reward

    # def is_duplicate_mask(self, center_x, center_y, size):
    #     for x, y, s in self.masked_regions:
    #         if abs(x - center_x) < 10 and abs(y - center_y) < 10:  # 位置接近
    #             return True
    #     return False
    def is_duplicate_mask(self, center_x, center_y, size):
        for x, y, s in self.masked_regions:
            # 计算重叠区域的边界
            overlap_x1 = max(center_x - size // 2, x - s // 2)
            overlap_x2 = min(center_x + size // 2, x + s // 2)
            overlap_y1 = max(center_y - size // 2, y - s // 2)
            overlap_y2 = min(center_y + size // 2, y + s // 2)

            # 计算重叠面积
            overlap_width = max(0, overlap_x2 - overlap_x1)
            overlap_height = max(0, overlap_y2 - overlap_y1)
            overlap_area = overlap_width * overlap_height

            # 计算两个 mask 的面积
            current_mask_area = size * size
            previous_mask_area = s * s

            # 计算相似度 (重叠面积 / 当前 mask 面积)
            similarity = overlap_area / max(current_mask_area, previous_mask_area)

            # 如果重叠面积超过 80%，认为是重复 mask
            if similarity >= 0.95:
                return True
        return False



# def metric_func(image,text_raw):#！！！！！！！！！！！！！！！！！一次只处理一张图片

#     return semantics, fluency 
    

# image_paths = ["path_to_image_1.jpg"]
# image_dataset = load_image_dataset(image_paths)#tesor


# env = TriggerRemovalEnv(image_dataset, metric_func)


# 6. 创建PPO模型
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)



