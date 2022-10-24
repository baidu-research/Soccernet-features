import glob
import subprocess
import datetime
import os
import argparse

def main(args):
    # sample filename /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2016-2017/2016-08-20 - 19-15 Barcelona 6 - 2 Betis/1_HQ.mkv
    files = sorted(glob.glob(os.path.join(args.input_folder, '**/*_HQ.mkv'), recursive= True))
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # import ipdb;ipdb.set_trace()

    for filename in files:
        # make necessary folders
        parts = filename.split('/')
        new_shortname_root = '.'.join(parts[-4:])
        new_filename = os.path.join(args.output_folder, new_shortname_root).replace(" ", "_").replace('HQ', 'LQ')
        
        command = f'ffmpeg -i "{filename}" -vf scale=456x256 -map 0:v -c:v libx264 -c:a aac "{new_filename}"'

        print(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default = '/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data')
    parser.add_argument('--output_folder', type=str, default = '/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference')
    parser.add_argument('--extension', type=str, default = 'mkv')


    args = parser.parse_args()
    main(args)
