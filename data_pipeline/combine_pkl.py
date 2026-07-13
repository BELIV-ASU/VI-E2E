import pickle
import argparse

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def find_closest_infra_cam(vehicle_ts, infra_infos,vehicle_scene):
    min_diff = float('inf')
    closest_cam = None

    for info in infra_infos:
        if info.get('scene_token') != vehicle_scene:
            continue
        cams = info.get('cams', {})
        cam = cams.get('INFRASTRUCTURE_CAM_V1', None)
        if cam is None:
            continue
        infra_ts = cam['timestamp']
        diff = abs(vehicle_ts - infra_ts)
        if diff < min_diff:
            min_diff = diff
            closest_cam = cam

    return closest_cam

def combine_vehicle_and_infra(vehicle_path, infra_path, output_path):
    vehicle_data = load_pkl(vehicle_path)
    infra_data = load_pkl(infra_path)

    vehicle_infos = vehicle_data['infos']
    infra_infos = infra_data['infos']

    print(f"Loaded {len(vehicle_infos)} vehicle entries")
    print(f"Loaded {len(infra_infos)} infrastructure entries")

    for idx, vehicle_sample in enumerate(vehicle_infos):
        vehicle_ts = vehicle_sample['cams']['VEHICLE_CAM_FRONT']['timestamp']
        vehicle_scene = vehicle_sample['scene_token']
        closest_cam = find_closest_infra_cam(vehicle_ts, infra_infos, vehicle_scene)

        if closest_cam is not None:
            if 'cams' not in vehicle_sample:
                vehicle_sample['cams'] = {}
            vehicle_sample['cams']['INFRASTRUCTURE_CAM_V1'] = closest_cam
        else:
            print(f"[Warning] No infra cam found for vehicle sample {idx}")

    out = {
        'infos': vehicle_infos,
        'metadata': vehicle_data.get('metadata', {})
    }
    save_pkl(out, output_path)
    print(f"Saved combined PKL to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle_pkl', required=True)
    parser.add_argument('--infra_pkl', required=True)
    parser.add_argument('--output_pkl', required=True)
    args = parser.parse_args()

    combine_vehicle_and_infra(args.vehicle_pkl, args.infra_pkl, args.output_pkl)

