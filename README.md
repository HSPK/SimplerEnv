# SIMPLER ENV MODIFIED VERSION

This repo is adapted from [SimplerEnv](https://github.com/simpler-env/SimplerEnv) to support parallel environments easily. Tested on A100 GPU with CUDA 12.2 and NVIDIA driver 535.

# Setup

1. Add to your dependencies
2. Download ManiSkill2_real2sim data

```
uv add "simpler_env@git+ssh://git@github.com/hspk/SimplerEnv.git@main"
git clone https://github.com/hspk/ManiSkill2_real2sim
```

3. Prepend the environment variable and run your command

```
MS2_REAL2SIM_ASSET_DIR=./ManiSkill2_real2sim/data <your command here>
```

# Usage

See [test_vector_env.py](./test_vector_env.py) for an example of using parallel environments.

Definition of `make_wrapped` function to create a vector environment:
```
env = make_vector(
    task_name = "YOUR_TASK_NAME",
    num_envs = 1,
    obs_angle_encoding = "euler",
    act_angle_encoding = "euler",
    google_robot_sticky_gripper_num_repeat = 15,
):

obs, info = env.reset()
print(obs["image"]["primary"].shape)
print(obs["instruction"])
print(obs["proprio"].shape)
```

# Known Issues

1. Can't load the `spaien` renderer when using `cuda` before creating the environment.

> You can create the environment first, then load the model or something that uses cuda.

2. `CUDA_VISIBLE_DEVICES` won't work.

> Haven't been solved yet.

3. Failed create simpler environment (spaien vulkan error) in some containered environments.

> Haven't been solved yet.
