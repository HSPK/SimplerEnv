# SIMPLER ENV MODIFIED VERSION

This repo is adapted from [SimplerEnv](https://github.com/simpler-env/SimplerEnv) to support parallel environments easily.

# Usage

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
