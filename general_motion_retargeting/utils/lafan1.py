import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def _alias_spine2(result):
    """Map non-LAFAN skeleton spine names to Spine2 expected by IK configs."""
    if "Spine2" in result:
        return
    if "Spine1" in result:
        result["Spine2"] = [result["Spine1"][0], result["Spine1"][1]]
    elif "Spine" in result:
        result["Spine2"] = [result["Spine"][0], result["Spine"][1]]


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
            
        if format == "lafan1":
            # LAFAN1: LeftToe/RightToe; some datasets: LeftToeBase/RightToeBase
            left_toe = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe = "RightToe" if "RightToe" in result else "RightToeBase"
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe][1]]
            _alias_spine2(result)
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
            _alias_spine2(result)
        elif format == "noitom":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
            _alias_spine2(result)
        elif format == "mocap":
            # FBX-like foot handling: foot position + foot orientation.
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


