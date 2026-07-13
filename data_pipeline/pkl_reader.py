import pickle
import argparse
from pprint import pprint
import json

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description="Read and print info from SPD/NuScenes pkl")
    parser.add_argument('--pkl_path', type=str, default='vad_nuscenes_infos_temporal_train_updated.pkl', help="Path to the pkl file")
    parser.add_argument('--index', type=int, default=0, help="Index of the info to inspect")
    args = parser.parse_args()

    data = load_pkl(args.pkl_path)
    infos = data['infos']
    metadata = data.get('metadata', {})
    print(len(infos))

    print("------ Metadata ------")
    pprint(metadata)

    print("\n------ Sample Info [index: {}] ------".format(args.index))
    pprint(infos[args.index])
    


if __name__ == '__main__':
    main()
