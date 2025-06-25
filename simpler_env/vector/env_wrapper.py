import gymnasium as gym
import numpy as np

from typing import TYPE_CHECKING


from simpler_env.utils.rotation_np import (
    euler_to_quaternion,
    euler_to_rotation_6d,
    make_unique_quaternion,
    matrix_to_euler,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_euler,
    quaternion_to_matrix,
    quaternion_to_rotation_6d,
    euler_to_axis_angle,
    quaternion_to_axis_angle,
    rotation_6d_to_axis_angle,
    axangle_to_euler,
    axis_angle_to_quaternion,
    axangle_to_rotation_6d,
)

if TYPE_CHECKING:
    from .async_vector_env import AsyncVectorEnv


def get_image(
    obs_image,
    robot_type,
    camera_name=None,
):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if robot_type == "google":
            camera_name = "overhead_camera"
        elif robot_type == "widowx":
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs_image[camera_name]["rgb"]


def encode_angle(
    x: np.ndarray, encoding: str, target_encoding: str = "euler"
) -> np.ndarray:
    fn_mapping = {
        ("euler", "euler"): (3, lambda x: x),
        ("euler", "quaternion"): (3, euler_to_quaternion),
        ("euler", "rot6d"): (3, euler_to_rotation_6d),
        ("euler", "axis_angle"): (3, euler_to_axis_angle),
        ("quaternion", "euler"): (4, quaternion_to_euler),
        ("quaternion", "quaternion"): (4, make_unique_quaternion),
        ("quaternion", "rot6d"): (4, quaternion_to_rotation_6d),
        ("quaternion", "axis_angle"): (4, quaternion_to_axis_angle),
        ("matrix", "euler"): (9, matrix_to_euler),
        ("matrix", "quaternion"): (9, matrix_to_quaternion),
        ("matrix", "rot6d"): (9, matrix_to_rotation_6d),
        ("matrix", "axis_angle"): (9, rotation_6d_to_axis_angle),
        ("axis_angle", "euler"): (3, axangle_to_euler),
        ("axis_angle", "quaternion"): (3, axis_angle_to_quaternion),
        ("axis_angle", "rot6d"): (3, axangle_to_rotation_6d),
        ("axis_angle", "axis_angle"): (3, lambda x: x),
    }
    _, fn = fn_mapping[(encoding, target_encoding)]

    return fn(x)


def preprocess_bridge_obs(
    obs: dict,
    default_rot: np.ndarray = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
    angle_encoding: str = "euler",
):
    # convert wxyz quat to xyzw quat
    proprio = obs["agent"]["eef_pos"]
    rot = np.roll(proprio[..., 3:-1], -1, axis=-1)  # wxyz to xyzw
    rm_bridge = quaternion_to_matrix(rot) @ default_rot.T
    gripper_openness = proprio[..., -1:]
    raw_proprio = np.concatenate(
        [
            proprio[..., :3],
            encode_angle(rm_bridge, "matrix", angle_encoding),
            gripper_openness,
        ],
        axis=-1,
    )
    obs["proprio"] = raw_proprio
    return obs


def postprocess_bridge_gripper(action: np.ndarray) -> np.ndarray:
    """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 close, 1 open for simpler
    action_gripper = 2.0 * (action > 0.5) - 1.0
    return action_gripper


def postprocess_single_google_gripper(
    action: float,
    sticky_action_is_on: bool,
    sticky_gripper_action: float,
    gripper_action_repeat: int,
    sticky_gripper_num_repeat: int = 15,
):
    """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 open, 1 close for simpler

    action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

    # without sticky
    relative_gripper_action = -action
    # if self.previous_gripper_action is None:
    #     relative_gripper_action = -1  # open
    # else:
    #     relative_gripper_action = -action
    # self.previous_gripper_action = action

    # switch to sticky closing
    if np.abs(relative_gripper_action) > 0.5 and sticky_action_is_on is False:
        sticky_action_is_on = True
        sticky_gripper_action = relative_gripper_action

    # sticky closing
    if sticky_action_is_on:
        gripper_action_repeat += 1
        relative_gripper_action = sticky_gripper_action

    # reaching maximum sticky
    if gripper_action_repeat == sticky_gripper_num_repeat:
        sticky_action_is_on = False
        gripper_action_repeat = 0
        sticky_gripper_action = 0.0

    return (
        relative_gripper_action,
        sticky_action_is_on,
        sticky_gripper_action,
        gripper_action_repeat,
    )


def postprocess_google_gripper(
    action: np.ndarray,
    sticky_action_is_on: list[bool],
    sticky_gripper_action: list[float],
    gripper_action_repeat: list[int],
    sticky_gripper_num_repeat: list[int],
):
    results = [
        postprocess_single_google_gripper(
            action[i],
            sticky_action_is_on[i],
            sticky_gripper_action[i],
            gripper_action_repeat[i],
            sticky_gripper_num_repeat[i],
        )
        for i in range(len(action))
    ]
    gripper_action = np.array([res[0] for res in results])
    sticky_action_is_on = [res[1] for res in results]
    sticky_gripper_action = [res[2] for res in results]
    gripper_action_repeat = [res[3] for res in results]
    return (
        gripper_action,
        sticky_action_is_on,
        sticky_gripper_action,
        gripper_action_repeat,
    )


def preprocess_google_obs(
    obs: dict,
    inverse_gripper: bool = False,
    angle_encoding: str = "euler",
) -> np.array:
    """convert wxyz quat from simpler to xyzw used in fractal"""
    quat_xyzw = np.roll(obs["agent"]["eef_pos"][..., 3:-1], -1, axis=-1)
    # from simpler, 0 for close, 1 for open
    gripper_width = obs["agent"]["eef_pos"][..., -1:]
    if not inverse_gripper:
        # we use 0 for close, 1 for open
        gripper_closedness = gripper_width
    else:
        gripper_closedness = 1 - gripper_width
    raw_proprio = np.concatenate(
        (
            obs["agent"]["eef_pos"][..., :3],
            encode_angle(quat_xyzw, "quaternion", angle_encoding),
            gripper_closedness,
        ),
        axis=-1,
    )
    obs["proprio"] = raw_proprio
    return obs


class SimplerVectorEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        task_name: str,
        env: "AsyncVectorEnv",
        num_envs: int = 1,
        obs_angle_encoding: str = "euler",
        act_angle_encoding: str = "euler",
        google_robot_sticky_gripper_num_repeat: int = 15,
    ):
        super().__init__(env)
        self.robot_type = "google" if "google_robot" in task_name else "widowx"
        self.obs_angle_encoding = obs_angle_encoding
        self.act_angle_encoding = act_angle_encoding
        self.num_envs = num_envs

        # Google Robot Specific Settings
        self.sticky_gripper_num_repeat = [
            google_robot_sticky_gripper_num_repeat
        ] * num_envs
        self.sticky_action_is_on = [False] * num_envs
        self.gripper_action_repeat = [0] * num_envs
        self.sticky_gripper_action = [0.0] * num_envs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.set_additional_info(obs)

        self.sticky_action_is_on = [False] * self.num_envs
        self.gripper_action_repeat = [0] * self.num_envs
        self.sticky_gripper_action = [0.0] * self.num_envs
        return obs, info

    def step(self, action: np.ndarray):
        action = self.prepare_env_action(action)
        return self.step_env_action(action)

    def step_env_action(self, env_action: np.ndarray):
        obs, rew, terminated, truncated, info = self.env.step(env_action)
        self.set_additional_info(obs)
        return obs, rew, terminated, truncated, info

    def prepare_env_action(self, action: np.ndarray) -> np.ndarray:
        xyz = action[..., :3]
        rot = action[..., 3:-1]
        grp = action[..., -1]
        if self.robot_type == "google":
            (
                grp,
                self.sticky_action_is_on,
                self.sticky_gripper_action,
                self.gripper_action_repeat,
            ) = postprocess_google_gripper(
                grp,
                self.sticky_action_is_on,
                self.sticky_gripper_action,
                self.gripper_action_repeat,
                self.sticky_gripper_num_repeat,
            )
        else:
            grp = postprocess_bridge_gripper(grp)
        rot = encode_angle(rot, self.act_angle_encoding, "axis_angle")
        action = np.concatenate((xyz, rot, grp[:, None]), axis=-1)
        return action

    def set_additional_info(self, obs: dict):
        obs["image"]["primary"] = get_image(obs["image"], self.robot_type)
        if self.robot_type == "google":
            obs = preprocess_google_obs(obs, angle_encoding=self.obs_angle_encoding)
        else:
            obs = preprocess_bridge_obs(obs, angle_encoding=self.obs_angle_encoding)
        obs["instruction"] = self.env.call("get_language_instruction")


def wrap_simpler_env(env: gym.Env, **kwargs):
    """Wraps simpler_env to add get_state, fix step method, and other utilities.

    :param env: The environment to wrap.
    :type env: gym.Env
    """

    return SimplerVectorEnvWrapper(env, **kwargs)
