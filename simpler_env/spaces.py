import numpy as np
from gymnasium.spaces import Box, Dict
from collections import OrderedDict
from copy import deepcopy

GOOGLE_ACTION_SPACE = Box(
    np.array(
        [-1.0, -1.0, -1.0, -1.5707964, -1.5707964, -1.5707964, -1.0], dtype=np.float32
    ),  # lower bounds
    np.array(
        [1.0, 1.0, 1.0, 1.5707964, 1.5707964, 1.5707964, 1.0], dtype=np.float32
    ),  # upper bounds
    (7,),  # shape
    np.float32,  # dtype
)

GOOGLE_OBSERVATION_SPACE = Dict(
    OrderedDict(
        {
            "agent": Dict(
                OrderedDict(
                    {
                        "qpos": Box(-np.inf, np.inf, (11,), np.float32),
                        "qvel": Box(-np.inf, np.inf, (11,), np.float32),
                        "eef_pos": Box(-np.inf, np.inf, (8,), np.float64),
                        "controller": Dict(
                            OrderedDict(
                                {
                                    "gripper": Dict(
                                        OrderedDict(
                                            {
                                                "target_qpos": Box(
                                                    -np.inf, np.inf, (2,), np.float32
                                                )
                                            }
                                        )
                                    )
                                }
                            )
                        ),
                        "base_pose": Box(-np.inf, np.inf, (7,), np.float32),
                    }
                )
            ),
            "extra": Dict(OrderedDict({})),
            "camera_param": Dict(
                OrderedDict(
                    {
                        "base_camera": Dict(
                            OrderedDict(
                                {
                                    "extrinsic_cv": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "cam2world_gl": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "intrinsic_cv": Box(
                                        -np.inf, np.inf, (3, 3), np.float32
                                    ),
                                }
                            )
                        ),
                        "overhead_camera": Dict(
                            OrderedDict(
                                {
                                    "extrinsic_cv": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "cam2world_gl": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "intrinsic_cv": Box(
                                        -np.inf, np.inf, (3, 3), np.float32
                                    ),
                                }
                            )
                        ),
                    }
                )
            ),
            "image": Dict(
                OrderedDict(
                    {
                        "base_camera": Dict(
                            OrderedDict(
                                {
                                    "rgb": Box(0, 255, (128, 128, 3), np.uint8),
                                    "depth": Box(
                                        0.0, np.inf, (128, 128, 1), np.float32
                                    ),
                                    "Segmentation": Box(
                                        0, 4294967295, (128, 128, 4), np.uint32
                                    ),
                                }
                            )
                        ),
                        "overhead_camera": Dict(
                            OrderedDict(
                                {
                                    "rgb": Box(0, 255, (512, 640, 3), np.uint8),
                                    "depth": Box(
                                        0.0, np.inf, (512, 640, 1), np.float32
                                    ),
                                    "Segmentation": Box(
                                        0, 4294967295, (512, 640, 4), np.uint32
                                    ),
                                }
                            )
                        ),
                    }
                )
            ),
        }
    )
)

GOOGLE_OBSERVATION_SPACE_WITH_EXTRA = deepcopy(GOOGLE_OBSERVATION_SPACE)
GOOGLE_OBSERVATION_SPACE_WITH_EXTRA["extra"] = Dict(
    OrderedDict({"tcp_pose": Box(-np.inf, np.inf, (7,), np.float32)})
)

BRIDGE_ACTION_SPACE = Box(
    np.array(
        [-1.0, -1.0, -1.0, -1.5707964, -1.5707964, -1.5707964, -1.0], dtype=np.float32
    ),  # lower bounds
    np.array(
        [1.0, 1.0, 1.0, 1.5707964, 1.5707964, 1.5707964, 1.0], dtype=np.float32
    ),  # upper bounds
    (7,),  # shape
    np.float32,  # dtype
)

BRIDGE_OBSERVATION_SPACE = Dict(
    OrderedDict(
        {
            "agent": Dict(
                OrderedDict(
                    {
                        "qpos": Box(-np.inf, np.inf, (8,), np.float32),
                        "qvel": Box(-np.inf, np.inf, (8,), np.float32),
                        "eef_pos": Box(-np.inf, np.inf, (8,), np.float64),
                        "controller": Dict(
                            OrderedDict(
                                {
                                    "arm": Dict(
                                        OrderedDict(
                                            {
                                                "target_pose": Box(
                                                    -np.inf, np.inf, (7,), np.float32
                                                )
                                            }
                                        )
                                    )
                                }
                            )
                        ),
                        "base_pose": Box(-np.inf, np.inf, (7,), np.float32),
                    }
                )
            ),
            "extra": Dict(
                OrderedDict({"tcp_pose": Box(-np.inf, np.inf, (7,), np.float32)})
            ),
            "camera_param": Dict(
                OrderedDict(
                    {
                        "base_camera": Dict(
                            OrderedDict(
                                {
                                    "extrinsic_cv": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "cam2world_gl": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "intrinsic_cv": Box(
                                        -np.inf, np.inf, (3, 3), np.float32
                                    ),
                                }
                            )
                        ),
                        "3rd_view_camera": Dict(
                            OrderedDict(
                                {
                                    "extrinsic_cv": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "cam2world_gl": Box(
                                        -np.inf, np.inf, (4, 4), np.float32
                                    ),
                                    "intrinsic_cv": Box(
                                        -np.inf, np.inf, (3, 3), np.float32
                                    ),
                                }
                            )
                        ),
                    }
                )
            ),
            "image": Dict(
                OrderedDict(
                    {
                        "base_camera": Dict(
                            OrderedDict(
                                {
                                    "rgb": Box(0, 255, (128, 128, 3), np.uint8),
                                    "depth": Box(
                                        0.0, np.inf, (128, 128, 1), np.float32
                                    ),
                                    "Segmentation": Box(
                                        0, 4294967295, (128, 128, 4), np.uint32
                                    ),
                                }
                            )
                        ),
                        "3rd_view_camera": Dict(
                            OrderedDict(
                                {
                                    "rgb": Box(0, 255, (480, 640, 3), np.uint8),
                                    "depth": Box(
                                        0.0, np.inf, (480, 640, 1), np.float32
                                    ),
                                    "Segmentation": Box(
                                        0, 4294967295, (480, 640, 4), np.uint32
                                    ),
                                }
                            )
                        ),
                    }
                )
            ),
        }
    )
)


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

SPACES = {
    k: (GOOGLE_ACTION_SPACE, GOOGLE_OBSERVATION_SPACE)
    if "google_robot" in k
    else (BRIDGE_ACTION_SPACE, BRIDGE_OBSERVATION_SPACE)
    for k in ENVIRONMENTS
}

for k in [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
]:
    SPACES[k] = (
        GOOGLE_ACTION_SPACE,
        GOOGLE_OBSERVATION_SPACE_WITH_EXTRA,
    )
