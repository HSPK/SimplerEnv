from collections import defaultdict
import os
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
FONT_PATH = str(Path(__file__) / "fonts/UbuntuMono-R.ttf")

_rng = np.random.RandomState(0)
_palette = ((_rng.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette

def make_grid(
    images: np.ndarray | list[np.ndarray],
    margin: int = 0,
    spacing: int = 10,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if isinstance(images, list):
        images = np.stack(images)
    assert images.ndim == 4, "Input should be a 4D array (N, H, W, C)"
    num_images, height, width, channels = images.shape
    grid_size = int(np.ceil(np.sqrt(num_images)))
    grid_height = grid_size * height + (grid_size - 1) * spacing + 2 * margin
    grid_width = grid_size * width + (grid_size - 1) * spacing + 2 * margin

    grid = np.full((grid_height, grid_width, channels), background, dtype=images.dtype)
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        y_start = margin + row * (height + spacing)
        y_end = y_start + height
        x_start = margin + col * (width + spacing)
        x_end = x_start + width
        grid[y_start:y_end, x_start:x_end] = img
    return grid

def topil(image: np.ndarray | Image.Image, target_size: tuple[int, int] = None):
    from PIL import Image

    if isinstance(image, Image.Image):
        if target_size is not None:
            image = image.resize(target_size, Image.BILINEAR)
        return image
    if target_size is not None:
        image = Image.fromarray(image).resize(target_size, Image.BILINEAR)
    else:
        image = Image.fromarray(image)
    return image


def save_video(images: list[np.ndarray | Image.Image], filename: str, fps: int = 30):
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip([np.array(topil(img)) for img in images], fps=fps)
    clip.write_videofile(filename, codec="libx264", audio=False)


def write_video(path, images, fps=5):
    # images: list of numpy arrays
    root_dir = Path(path).parent
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not isinstance(images[0], np.ndarray):
        images_npy = [image.numpy() for image in images]
    else:
        images_npy = images
    media.write_video(path, images_npy, fps=fps)


def plot_pred_and_gt_action_trajectory(predicted_actions, gt_actions, stacked_images):
    """
    Plot predicted and ground truth action trajectory
    Args:
        predicted_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        gt_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        stacked_images: np.array, [H, W * n_images, 3], uint8 (here n_images does not need to be the same as the length of predicted_actions or gt_actions)
    """

    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)
    figure_layout = [
        "terminate_episode_0",
        "terminate_episode_1",
        "terminate_episode_2",
        "world_vector_0",
        "world_vector_1",
        "world_vector_2",
        "rotation_delta_0",
        "rotation_delta_1",
        "rotation_delta_2",
        "gripper_closedness_action_0",
    ]
    action_order = [
        "terminate_episode",
        "world_vector",
        "rotation_delta",
        "gripper_closedness_action",
    ]

    for i, action in enumerate(gt_actions):
        for action_name in action_order:
            for action_sub_dimension in range(action[action_name].shape[0]):
                # print(action_name, action_sub_dimension)
                title = f"{action_name}_{action_sub_dimension}"
                action_name_to_values_over_time[title].append(action[action_name][action_sub_dimension])
                predicted_action_name_to_values_over_time[title].append(
                    predicted_actions[i][action_name][action_sub_dimension]
                )

    figure_layout = [["image"] * len(figure_layout), figure_layout]

    plt.rcParams.update({"font.size": 12})

    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    for i, (k, v) in enumerate(action_name_to_values_over_time.items()):

        axs[k].plot(v, label="ground truth")
        axs[k].plot(predicted_action_name_to_values_over_time[k], label="predicted action")
        axs[k].set_title(k)
        axs[k].set_xlabel("Time in one episode")

    axs["image"].imshow(stacked_images)
    axs["image"].set_xlabel("Time in one episode (subsampled)")

    plt.legend()
    plt.show()
