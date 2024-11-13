import os
import math

import torch
from torchvision.io import write_video
from diffusers.utils import load_image
import imageio


def save_videos(videos, start_index, save_path, fps):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
        for i, video in enumerate(videos):
            save_path_i = os.path.join(save_path, f"video_{start_index + i}.mp4")
            write_video(save_path_i, video, fps=fps, video_codec="h264")
    elif videos.ndim == 4:
        save_path = os.path.join(save_path, f"video_{start_index}.mp4")
        write_video(save_path, video, fps=fps, video_codec="h264")
    else:
        raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")


def save_video_grid(videos, save_path, fps, nrow=None):
    b, t, h, w, c = videos.shape
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t,
            (padding + h) * nrow + padding,
            (padding + w) * ncol + padding,
            c
        ),
        dtype=torch.uint8
    )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r: start_r + h, start_c: start_c + w] = videos[i]
    
    imageio.mimwrite(os.path.join(save_path, "video_grid.mp4"), video_grid, fps=fps, quality=6)


def load_prompts(prompt):
    if os.path.exists(prompt):
        with open(prompt, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        return prompts
    else:
        return [prompt]


def load_images(image=None):
    if image is None:
        print("The input image is None, excute text to video task")
        return None
    
    if os.path.exists(image):
        if os.path.splitext(image)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
            return [load_image(image)]
        else:
            with open(image, "r") as f:
                images = [load_image(line.strip()) for line in f.readlines()]
            return images
    else:
        raise FileNotFoundError(f"The image path {image} does not exist")