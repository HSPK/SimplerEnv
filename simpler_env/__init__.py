import gymnasium as gym
import mani_skill2_real2sim.envs  # noqa: F401

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

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": (
        "PlaceIntoClosedDrawerCustomInScene-v0",
        {},
    ),
    "google_robot_place_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0",
        {},
    ),
    "google_robot_place_in_closed_middle_drawer": (
        "PlaceIntoClosedMiddleDrawerCustomInScene-v0",
        {},
    ),
    "google_robot_place_in_closed_bottom_drawer": (
        "PlaceIntoClosedBottomDrawerCustomInScene-v0",
        {},
    ),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0",
        {"model_ids": "baked_apple_v2"},
    ),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
}


def make(task_name: str, **kwargs):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, (
        f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    )
    env_name, env_kwargs = ENVIRONMENT_MAP[task_name]
    kwargs.update(env_kwargs)
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env


def make_wrapped(
    task_name: str,
    obs_angle_encoding: str = "euler",
    act_angle_encoding: str = "euler",
    google_robot_sticky_gripper_num_repeat: int = 15,
    **kwargs,
):
    from simpler_env.wrapper.env_wrapper import SimplerEnvWrapper

    return SimplerEnvWrapper(
        make(task_name, **kwargs),
        obs_angle_encoding=obs_angle_encoding,
        act_angle_encoding=act_angle_encoding,
        google_robot_sticky_gripper_num_repeat=google_robot_sticky_gripper_num_repeat,
    )


def make_vector(
    task_name: str,
    num_envs: int = 1,
    obs_angle_encoding: str = "euler",
    act_angle_encoding: str = "euler",
    google_robot_sticky_gripper_num_repeat: int = 15,
    **kwargs,
):
    from simpler_env.vector.async_vector_env import AsyncVectorEnv
    from simpler_env.vector.env_wrapper import SimplerVectorEnvWrapper
    from simpler_env.spaces import SPACES

    return SimplerVectorEnvWrapper(
        task_name=task_name,
        env=AsyncVectorEnv(
            [lambda: make(task_name, **kwargs)] * num_envs,
            action_space=SPACES[task_name][0],
            observation_space=SPACES[task_name][1],
        ),
        num_envs=num_envs,
        obs_angle_encoding=obs_angle_encoding,
        act_angle_encoding=act_angle_encoding,
        google_robot_sticky_gripper_num_repeat=google_robot_sticky_gripper_num_repeat,
    )
