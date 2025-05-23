import numpy as np
from PIL import Image


def generate_gaussian_noise_image(image_size, std, save_path):
    mean = 0
    gaussian_noise = np.random.normal(mean, std, (image_size, image_size, 3))
    gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)  # 限制像素值范围为 [0, 255]

    image = Image.fromarray(gaussian_noise)
    image.save(save_path)

    return gaussian_noise


image_size = 20 
std = 20       
save_path = "resource/TrojVLM/gaussian_noise_image.png"


gaussian_noise_image = generate_gaussian_noise_image(image_size, std, save_path)


print(f"Gaussian noise image saved at {save_path}")
