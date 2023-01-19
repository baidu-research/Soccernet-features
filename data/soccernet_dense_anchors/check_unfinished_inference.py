import glob
import os
import argparse
import numpy as np

def main(args):
    matches_to_do = set()

    if len(args.inference_dirs_file) > 0:
        with open(args.inference_dirs_file) as f:
            lines = f.readlines()
        for line in lines:
            if not os.path.exists(line.strip()) or (not os.path.exists(line.strip() + '/features.npy')):
                match_to_do = line.strip().split('/')[-1]
                # print(line)
                print(match_to_do)
    else:
        for subfolder in glob.glob(f'{args.inference_root}/*/'):
            features_filename = os.path.join(subfolder, 'features.npy')
            match_to_do = subfolder.split('/')[-2]

            # print(match_to_do)

            if not os.path.exists(features_filename):
                # print(subfolder)
                print(match_to_do)
            else:
                features = np.load(features_filename, allow_pickle = True)
                if not len(features.item().keys()) == 3:
                    # print('old features')
                    print(match_to_do)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_root', type=str, default = '/mnt/storage/gait-0/xin/soccernet_features', help = 'Root folder of inference results')
    parser.add_argument('--inference_dirs_file', type=str, default = '', help = 'Root folder of inference results')

    args = parser.parse_args()
    main(args)