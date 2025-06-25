from simpler_env.utils.visualization import save_video, make_grid, topil

from simpler_env import make_wrapped, make_vector
from simpler_env.spaces import BRIDGE_ACTION_SPACE, BRIDGE_OBSERVATION_SPACE
import warnings
import logging
import torch


svk_logger = logging.getLogger("svulkan2")
svk_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def vector_env_test():
    from pathlib import Path

    num_envs = 16
    for task in ENVIRONMENTS:
        print(f"Testing task: {task}")
        if Path(f"{task}_video.mp4").exists():
            print(f"Video for {task} already exists, skipping...")
            # continue
        env = make_vector(
            task,
            num_envs=num_envs,
            obs_angle_encoding="euler",
            act_angle_encoding="euler",
        )

        torch.tensor([1, 2, 3]).cuda()
        print("Use Cuda after make vector")

        seed = list(range(num_envs))
        images = []
        obs, info = env.reset(
            seed=seed,
            options=[{"obj_init_options": {"episode_id": i}} for i in range(num_envs)],
        )
        action = np.array([[0, 0, 0, 0, 0, 0.01, 0]] * num_envs)
        for _ in range(10):
            images.append(topil(make_grid(obs["image"]["primary"]), (1024, 1024)))
            obs, reward, terminated, truncated, info = env.step(action)
        images.append(topil(make_grid(obs["image"]["primary"]), (1024, 1024)))
        save_video(images, f"{task}_video.mp4", fps=10)

        env.close()
        del env


def single_env_test():
    env = make_wrapped(
        "widowx_stack_cube",
        obs_angle_encoding="euler",
        act_angle_encoding="euler",
    )
    assert env.action_space == BRIDGE_ACTION_SPACE
    from pprint import pprint

    space = env.observation_space
    pprint(space, width=120, compact=False)
    print(type(env.observation_space.spaces))
    assert env.observation_space == BRIDGE_OBSERVATION_SPACE


if __name__ == "__main__":
    # single_env_test()
    vector_env_test()
