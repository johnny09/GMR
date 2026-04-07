import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def _alias_spine2_for_lafan_ik(result):
    """
    bvh_lafan1 IK expects a Spine2 body. LAFAN1 BVHs have Spine2; Mixamo / PND often have
    Hips -> Spine -> Spine1 -> Neck (no Spine2). Duplicate Spine1 (or Spine) pose as Spine2.
    """
    if "Spine2" in result:
        return
    if "Spine1" in result:
        src = "Spine1"
    elif "Spine" in result:
        src = "Spine"
    else:
        raise KeyError(
            "BVH has no Spine2 (LAFAN1). Add Spine1 or Spine (Mixamo/PND) for IK alias."
        )
    pos, rot = result[src]
    result["Spine2"] = [np.asarray(pos, dtype=np.float64).copy(), np.asarray(rot, dtype=np.float64).copy()]


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
            # LAFAN1: LeftToe / RightToe. Mixamo / PND mocap: LeftToeBase / RightToeBase.
            if "LeftToe" in result:
                left_toe_rot = result["LeftToe"][1]
            elif "LeftToeBase" in result:
                left_toe_rot = result["LeftToeBase"][1]
            else:
                raise KeyError("Need LeftToe (LAFAN1) or LeftToeBase (Mixamo/PND) in BVH")
            if "RightToe" in result:
                right_toe_rot = result["RightToe"][1]
            elif "RightToeBase" in result:
                right_toe_rot = result["RightToeBase"][1]
            else:
                raise KeyError("Need RightToe (LAFAN1) or RightToeBase (Mixamo/PND) in BVH")
            result["LeftFootMod"] = [result["LeftFoot"][0], left_toe_rot]
            result["RightFootMod"] = [result["RightFoot"][0], right_toe_rot]
            _alias_spine2_for_lafan_ik(result)
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
            _alias_spine2_for_lafan_ik(result)
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


