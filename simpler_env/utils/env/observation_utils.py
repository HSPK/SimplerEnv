def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    import torch

    # obtain image from observation dictionary returned by ManiSkill3 environment
    if camera_name is None:
        if "google_robot" in env.unwrapped.robot_uids.uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uids.uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["sensor_data"][camera_name]["rgb"].to(torch.uint8)
