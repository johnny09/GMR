import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
        help="BVH format type.",
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
            "pnd_adam_lite",
            "pnd_adam_sp",
        ],
        default="unitree_g1",
    )

    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override existing files.",
    )

    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
        help="Target FPS for the retargeted motion.",
    )

    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed npz format to reduce file size.",
    )

    parser.add_argument(
        "--compute_local_body_pos",
        action="store_true",
        default=False,
        help="Compute local body positions using forward kinematics (requires GPU).",
    )

    parser.add_argument(
        "--height_adjust",
        action="store_true",
        default=False,
        help="Adjust root height to prevent ground penetration.",
    )

    parser.add_argument(
        "--perframe_adjust",
        action="store_true",
        default=False,
        help="Adjust height per frame (only works with --height_adjust).",
    )

    args = parser.parse_args()

    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    # Create target folder if it doesn't exist
    os.makedirs(tgt_folder, exist_ok=True)

    # Collect all BVH files
    bvh_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if filename.endswith(".bvh"):
                bvh_file_path = os.path.join(dirpath, filename)
                bvh_files.append(bvh_file_path)

    print(f"Found {len(bvh_files)} BVH files to process")

    # Process each BVH file
    for bvh_file_path in tqdm(bvh_files, desc="Retargeting files"):
        # Get the target file path
        relative_path = os.path.relpath(bvh_file_path, src_folder)
        tgt_file_path = os.path.join(tgt_folder, relative_path).replace(".bvh", ".npz")

        if os.path.exists(tgt_file_path) and not args.override:
            print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
            continue

        # Load BVH trajectory
        try:
            bvh_data_frames, actual_human_height = load_bvh_file(
                bvh_file_path, format=args.format
            )
            src_fps = args.target_fps
        except Exception as e:
            print(f"Error loading {bvh_file_path}: {e}")
            continue

        # Initialize the retargeting system
        try:
            retargeter = GMR(
                src_human=f"bvh_{args.format}",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retargeter.xml_file)
            data = mj.MjData(model)
        except Exception as e:
            print(f"Error initializing retargeter for {bvh_file_path}: {e}")
            continue

        # Retarget to get all qpos
        qpos_list = []
        qvel_list = []
        try:
            for curr_frame in range(len(bvh_data_frames)):
                smplx_data = bvh_data_frames[curr_frame]

                # Retarget till convergence
                qpos, qvel = retargeter.retarget(smplx_data)

                qpos_list.append(qpos.copy())
                qvel_list.append(qvel.copy())
        except Exception as e:
            print(f"Error retargeting {bvh_file_path}: {e}")
            continue

        qpos_list = np.array(qpos_list)
        qvel_list = np.array(qvel_list)
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]  # Keep wxyz format
        dof_pos = qpos_list[:, 7:]
        dof_vel = qvel_list[:, 6:]
        num_frames = root_pos.shape[0]

        # Compute local body positions if requested
        local_body_pos = None
        body_names = None

        if args.compute_local_body_pos:
            try:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

                # Obtain local body pos
                identity_root_pos = torch.zeros((num_frames, 3), device=device)
                identity_root_rot = torch.zeros((num_frames, 4), device=device)
                identity_root_rot[:, 0] = 1.0  # wxyz format: set w=1
                local_body_pos, _ = kinematics_model.forward_kinematics(
                    identity_root_pos,
                    identity_root_rot,
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
                )
                body_names = kinematics_model.body_names

                # Height adjustment if requested
                if args.height_adjust:
                    body_pos, _ = kinematics_model.forward_kinematics(
                        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
                    )
                    ground_offset = 0.00
                    if not args.perframe_adjust:
                        lowest_height = torch.min(body_pos[..., 2]).item()
                        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                    else:
                        for i in range(root_pos.shape[0]):
                            lowest_body_part = torch.min(body_pos[i, :, 2])
                            root_pos[i, 2] = (
                                root_pos[i, 2] - lowest_body_part + ground_offset
                            )

                # Convert to numpy
                local_body_pos = local_body_pos.detach().cpu().numpy()

            except Exception as e:
                print(
                    f"Warning: Error computing local body pos for {bvh_file_path}: {e}"
                )
                local_body_pos = None
                body_names = None

        # Prepare data dictionary for npz format
        save_dict = {
            "fps": np.array([src_fps]),  # Convert to array for npz
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }

        # Only add optional fields if they are not None
        if local_body_pos is not None:
            save_dict["local_body_pos"] = local_body_pos
        if body_names is not None:
            save_dict["link_body_list"] = body_names

        # Create target directory if needed
        os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)

        # Save as npz or compressed npz
        try:
            if args.compressed:
                np.savez_compressed(tgt_file_path, **save_dict)
            else:
                np.savez(tgt_file_path, **save_dict)
        except Exception as e:
            print(f"Error saving {tgt_file_path}: {e}")
            continue

    print(f"Done. Saved {len(bvh_files)} files to {tgt_folder}")
