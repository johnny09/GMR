import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import os
import joblib
import numpy as np

try:
    from mink.exceptions import NotWithinConfigurationLimits
except ImportError:
    NotWithinConfigurationLimits = Exception  # fallback if mink structure changes

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="Single BVH motion file to load. Ignored if --src_folder is set.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--src_folder",
        help="Directory of BVH files to convert; all will be merged into one pkl.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
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
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--T",
        type=int,
        default=None,
        dest="clip_frames",
        help="Clip length in frames. If not set, save the entire motion as one clip.",
    )

    args = parser.parse_args()

    if args.src_folder is not None and args.bvh_file is not None:
        raise ValueError("Use either --bvh_file or --src_folder, not both.")
    if args.src_folder is None and args.bvh_file is None:
        raise ValueError("Provide either --bvh_file or --src_folder.")
    if args.save_path is None:
        raise ValueError("--save_path is required.")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ndof = 29  # BFM format
    motion_fps = args.motion_fps

    def build_clips_from_qpos_list(qpos_list, model, motion_name, clip_frames):
        """Build clip dict from qpos_list. Returns dict of clip_key -> clip."""
        total_frames = len(qpos_list)
        if clip_frames is None:
            segments = [qpos_list]
        else:
            segments = [
                qpos_list[start : start + clip_frames]
                for start in range(0, total_frames, clip_frames)
            ]
            segments = [s for s in segments if len(s) > 0]

        def build_clip(qpos_slice):
            root_trans_offset = np.array([qpos[:3] for qpos in qpos_slice], dtype=np.float32)
            root_rot = np.array(
                [np.array(qpos[3:7])[[1, 2, 3, 0]] for qpos in qpos_slice],
                dtype=np.float32,
            )
            dof = np.array(
                [qpos[7 : 7 + ndof] for qpos in qpos_slice],
                dtype=np.float32,
            )
            pose_aa_list = []
            for qpos in qpos_slice:
                root_quat_wxyz = np.asarray(qpos[3:7])
                root_aa = R.from_quat(
                    root_quat_wxyz[[1, 2, 3, 0]], scalar_first=False
                ).as_rotvec()
                joint_aa = []
                for k in range(ndof):
                    jnt_id = model.dof_jntid[6 + k]
                    axis = np.asarray(model.jnt_axis[jnt_id], dtype=np.float64)
                    angle = float(qpos[7 + k])
                    joint_aa.append(axis * angle)
                pose_aa_list.append(
                    np.concatenate([root_aa[np.newaxis, :], np.array(joint_aa)], axis=0)
                )
            pose_aa = np.array(pose_aa_list, dtype=np.float32)
            return {
                "root_trans_offset": root_trans_offset,
                "pose_aa": pose_aa,
                "dof": dof,
                "root_rot": root_rot,
                "fps": motion_fps,
                "motion_name": motion_name,
            }

        # Top-level key: unique clip ID like "walk1_subject5_clip13", "fallAndGetUp1_subject4_clip0"
        return {
            f"{motion_name}_clip{i}": build_clip(seg)
            for i, seg in enumerate(segments)
        }

    data = {}

    if args.src_folder is not None:
        # Batch: collect all BVH files under src_folder
        bvh_files = []
        for dirpath, _, filenames in os.walk(args.src_folder):
            for f in sorted(filenames):
                if f.endswith(".bvh"):
                    bvh_files.append(os.path.join(dirpath, f))
        print(f"Found {len(bvh_files)} BVH file(s) in {args.src_folder}")

        for bvh_file_path in tqdm(bvh_files, desc="Converting files"):
            try:
                lafan1_data_frames, actual_human_height = load_bvh_file(
                    bvh_file_path, format=args.format
                )
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue
            retargeter = GMR(
                src_human=f"bvh_{args.format}",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            qpos_list = []
            n_fallback = 0
            for smplx_data in lafan1_data_frames:
                try:
                    qpos, _ = retargeter.retarget(smplx_data)
                    qpos_list.append(qpos)
                except NotWithinConfigurationLimits:
                    if qpos_list:
                        qpos_list.append(qpos_list[-1].copy())
                        n_fallback += 1
            if not qpos_list:
                continue
            if n_fallback > 0:
                print(f"  [IK limits] {os.path.basename(bvh_file_path)}: {n_fallback} frame(s) used fallback pose")
            # Key format: "{motion_name}_clip{i}" e.g. "walk1_subject5_clip13", "fallAndGetUp1_subject4_clip0"
            motion_name = os.path.splitext(os.path.basename(bvh_file_path))[0]
            file_data = build_clips_from_qpos_list(
                qpos_list, retargeter.model, motion_name, args.clip_frames
            )
            # Ensure unique keys: if same basename in different folders, add _1, _2, ...
            disambiguator = 0
            while any(k in data for k in file_data):
                disambiguator += 1
                motion_name_unique = f"{motion_name}_{disambiguator}"
                file_data = {
                    f"{motion_name_unique}_clip{i}": {**v, "motion_name": motion_name_unique}
                    for i, (k, v) in enumerate(file_data.items())
                }
            data.update(file_data)

        joblib.dump(data, args.save_path)
        print(f"Saved to {args.save_path} ({len(data)} clip(s) from {len(bvh_files)} file(s))")

    else:
        # Single file (with optional viewer)
        lafan1_data_frames, actual_human_height = load_bvh_file(
            args.bvh_file, format=args.format
        )
        retargeter = GMR(
            src_human=f"bvh_{args.format}",
            tgt_robot=args.robot,
            actual_human_height=actual_human_height,
        )
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
        )
        print(f"mocap_frame_rate: {motion_fps}")
        pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0
        i = 0
        qpos_list = []

        while True:
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
            pbar.update(1)
            smplx_data = lafan1_data_frames[i]
            try:
                qpos, qvel = retargeter.retarget(smplx_data)
                qpos_list.append(qpos)
            except NotWithinConfigurationLimits:
                if qpos_list:
                    qpos = qpos_list[-1].copy()
                    qpos_list.append(qpos)
                else:
                    i += 1
                    if i >= len(lafan1_data_frames):
                        break
                    continue
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    break
        pbar.close()
        robot_motion_viewer.close()

        motion_name = os.path.splitext(os.path.basename(args.bvh_file))[0]
        data = build_clips_from_qpos_list(
            qpos_list, retargeter.model, motion_name, args.clip_frames
        )
        joblib.dump(data, args.save_path)
        print(f"Saved to {args.save_path} ({len(data)} clip(s))")
