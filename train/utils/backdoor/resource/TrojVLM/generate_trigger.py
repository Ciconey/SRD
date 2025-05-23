'''This script is to generate a black image with only a white square at the right corner, then convert it to a npy file'''

import numpy as np
import argparse
from PIL import Image


def generate_gaussian_trigger_image(image_size, trigger_size, std, distance_to_right, distance_to_bottom):
    mean = 0
    gaussian_trigger = np.random.normal(mean, std, (trigger_size, trigger_size, 3))
    gaussian_trigger = np.clip(gaussian_trigger, 0, 255).astype(np.uint8)  # 限制到

    black_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    x_start = np.random.randint(0, image_size - trigger_size)
    y_start = np.random.randint(0, image_size - trigger_size)
    black_image[x_start:x_start + trigger_size, y_start:y_start + trigger_size, :] = gaussian_trigger

    return black_image


def generate_trigger_mask(image_size, trigger_size, std):

    gaussian_trigger = np.ones((20, 20, 3), dtype=np.uint8)
    gaussian_trigger = gaussian_trigger / 255

    black_image = np.zeros((image_size, image_size, 3), dtype=np.float32)

    black_image[0:trigger_size, 0:trigger_size, :] = gaussian_trigger
    black_image = np.round(black_image).astype(np.uint8)
    return black_image


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--image_size', type=int, default=224)
    args.add_argument('--patch_size', type=int, default=20)
    args.add_argument('--std', type=int, default=20)
    args.add_argument('--distance_to_right', type=int, default=204)
    args.add_argument('--distance_to_bottom', type=int, default=204)
    args.add_argument('--output_path', type=str, default='/resource/TrojVLM/trigger_test.png')
    args = args.parse_args()
    image = generate_gaussian_trigger_image(
        args.image_size,
        args.patch_size,
        args.std,
        args.distance_to_right,
        args.distance_to_bottom,
        # args.patch_path,
        # args.distance_to_right,
        # args.distance_to_bottom,
    )

    Image.fromarray(image).save(args.output_path)
    # Image.fromarray(mask).save(args.output_path)
