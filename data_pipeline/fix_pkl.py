import os
import json
import pickle
import numpy as np

# === Paths ===
PKL_PATH = "V/vad_nuscenes_infos_temporal_train.pkl"
UPDATED_PKL_PATH = "V/vad_nuscenes_infos_temporal_train_updated.pkl"
SAMPLE_DATA_PATH = "V/v1.0-trainval/sample_data.json"
CALIB_SENSOR_PATH = "V/v1.0-trainval/calibrated_sensor.json"
EGO_POSE_PATH = "V/v1.0-trainval/ego_pose.json"
SENSOR_PATH = "V/v1.0-trainval/sensor.json"

# === Load data ===
with open(PKL_PATH, "rb") as f:
    pkl_data = pickle.load(f)

with open(SAMPLE_DATA_PATH, "r") as f:
    sample_data = json.load(f)

with open(CALIB_SENSOR_PATH, "r") as f:
    calibrated_sensors = json.load(f)

with open(EGO_POSE_PATH, "r") as f:
    ego_poses = json.load(f)

with open(SENSOR_PATH, "r") as f:
    sensors = json.load(f)

# === Build lookup maps ===
sensor_token_to_channel = {s["token"]: s["channel"] for s in sensors}
calib_sensor_map = {c["token"]: c for c in calibrated_sensors}
ego_pose_map = {e["token"]: e for e in ego_poses}

# Build mapping: (sample_token, channel) -> sample_data
sample_data_by_sample_token_and_channel = {}

for sd in sample_data:
    calib_token = sd["calibrated_sensor_token"]
    cs_entry = calib_sensor_map.get(calib_token)
    if cs_entry is None:
        continue
    sensor_token = cs_entry["sensor_token"]
    channel = sensor_token_to_channel.get(sensor_token)
    if channel is None:
        continue
    key = (sd["sample_token"], channel)
    sample_data_by_sample_token_and_channel[key] = sd

# === Helper: Compute sensor-to-lidar transformation using rotation matrices directly ===
def to_rotation_matrix(rot):
    rot = np.array(rot)
    if rot.shape == (3, 3):
        return rot
    elif rot.shape == (4,):
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(rot).as_matrix()
    elif rot.shape == (4, 4):  # homogeneous matrix
        return rot[:3, :3]
    else:
        raise ValueError(f"Unexpected rotation shape: {rot.shape}")

def to_translation_vector(t):
    t = np.array(t)
    if t.shape == (4,):
        return t[:3]
    return t

def compute_sensor_to_lidar(cs_cam, ep_cam, cs_lidar, ep_lidar):
    R_lidar = to_rotation_matrix(cs_lidar["rotation"])
    T_lidar = to_translation_vector(cs_lidar["translation"])
    R_ego_lidar = to_rotation_matrix(ep_lidar["rotation"])
    T_ego_lidar = to_translation_vector(ep_lidar["translation"])

    R_cam = to_rotation_matrix(cs_cam["rotation"])
    T_cam = to_translation_vector(cs_cam["translation"])
    R_ego_cam = to_rotation_matrix(ep_cam["rotation"])
    T_ego_cam = to_translation_vector(ep_cam["translation"])

    cam_in_world = R_ego_cam @ R_cam
    lidar_in_world = R_ego_lidar @ R_lidar
    R_sensor2lidar = np.linalg.inv(lidar_in_world) @ cam_in_world

    t_cam_global = R_ego_cam @ T_cam + T_ego_cam
    t_lidar_global = R_ego_lidar @ T_lidar + T_ego_lidar
    T_sensor2lidar = np.linalg.inv(lidar_in_world) @ (t_cam_global - t_lidar_global)

    return R_sensor2lidar, T_sensor2lidar


# === Main update loop ===
updated_count = 0
camera_type = "VEHICLE_CAM_FRONT"

for info in pkl_data["infos"]:
    sample_token = info["token"]
    key = (sample_token, camera_type)
    if key not in sample_data_by_sample_token_and_channel:
        continue

    sd = sample_data_by_sample_token_and_channel[key]
    calib_token = sd["calibrated_sensor_token"]
    ego_token = sd["ego_pose_token"]

    cs_cam = calib_sensor_map.get(calib_token)
    ep_cam = ego_pose_map.get(ego_token)

    if cs_cam is None or ep_cam is None:
        continue

    cs_lidar = {
        "rotation": info["lidar2ego_rotation"],
        "translation": info["lidar2ego_translation"]
    }
    ep_lidar = {
        "rotation": info["ego2global_rotation"],
        "translation": info["ego2global_translation"]
    }

    R_lidar, T_lidar = compute_sensor_to_lidar(cs_cam, ep_cam, cs_lidar, ep_lidar)

    cam_entry = info["cams"].get(camera_type, {})

    cam_entry["data_path"] = "/home/aiyer43/UniV2X/datasets/V2X-Seq-SPD-New/vehicle-side/"+sd["filename"]
    cam_entry["sensor2ego_translation"] = cs_cam["translation"]
    cam_entry["sensor2ego_rotation"] = cs_cam["rotation"]
    cam_entry["ego2global_translation"] = ep_cam["translation"]
    cam_entry["ego2global_rotation"] = ep_cam["rotation"]
    cam_entry["timestamp"] = sd["timestamp"]
    cam_entry["cam_intrinsic"] = cs_cam.get("camera_intrinsic", [])
    cam_entry["type"] = camera_type
    cam_entry["sample_data_token"] = sd["token"]

    cam_entry["sensor2lidar_rotation"] = np.array(R_lidar.T.tolist())
    cam_entry["sensor2lidar_translation"] = np.array(T_lidar.tolist())

    info["cams"][camera_type] = cam_entry
    updated_count += 1

# === Save updated PKL ===
with open(UPDATED_PKL_PATH, "wb") as f:
    pickle.dump(pkl_data, f)

print(f"[✓] Updated {updated_count} VEHICLE_CAM_FRONT entries with sensor-to-lidar transform.")
print(f"[📦] Saved to: {UPDATED_PKL_PATH}")

