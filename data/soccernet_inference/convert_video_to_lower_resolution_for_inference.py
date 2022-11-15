import glob
import subprocess
import datetime
import os
import argparse

def parse_gamestart_secs_line(line):
    # get video_ini
    # "/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/video.ini"
    # sample
    # [1_HQ.mkv]
    # start_time_second = 67

    # [2_HQ.mkv]
    # start_time_second = 52
    return int(line.split('=')[-1])

def main(args):
    # sample filename /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2016-2017/2016-08-20 - 19-15 Barcelona 6 - 2 Betis/1_HQ.mkv
    files = sorted(glob.glob(os.path.join(args.input_folder, '**/*_HQ.mkv'), recursive= True))
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # import ipdb;ipdb.set_trace()

    for filename in files:
        match_folder = os.path.dirname(filename)
        videos_starts_filename = os.path.join(match_folder, 'video.ini')

        with open(videos_starts_filename, 'r') as g:
            lines = g.readlines()
        game_start_secs_in_videos = [parse_gamestart_secs_line(lines[1]), parse_gamestart_secs_line(lines[4])]

        game_half = os.path.basename(filename).split('_')[0]
        game_half_index = int(game_half) - 1
        # offset and 45 min

        game_start_secs = game_start_secs_in_videos[game_half_index]
        # game_end_secs = game_start_secs + 45 * 60
        
        # make necessary folders
        parts = filename.split('/')
        new_shortname_root = '.'.join(parts[-4:])
        new_filename = os.path.join(args.output_folder, new_shortname_root).replace(" ", "_").replace('HQ', 'LQ')
        
        if args.trim_to_gametime:
            if args.fps > 0:
                command = f'ffmpeg -ss {game_start_secs} -i "{filename}" -vf "scale=456x256,fps={args.fps}" -map 0:v -c:v libx264 -strict experimental "{new_filename}" -y'
            else:
                command = f'ffmpeg -ss {game_start_secs} -i "{filename}" -vf scale=456x256 -map 0:v -c:v libx264 -c:a aac -strict experimental "{new_filename}"'            
        else:
            if args.fps > 0:
                command = f'ffmpeg -i "{filename}" -vf "scale=456x256,fps={args.fps}" -map 0:v -c:v libx264 "{new_filename}" -y'
            else:
                command = f'ffmpeg -i "{filename}" -vf scale=456x256 -map 0:v -c:v libx264 -c:a aac "{new_filename}"'

        print(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default = '/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data')
    parser.add_argument('--output_folder', type=str, default = '/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference')
    parser.add_argument('--fps', type=int, default = -1)
    parser.add_argument('--extension', type=str, default = 'mkv')

    parser.add_argument('--trim_to_gametime', dest='trim_to_gametime', action='store_true')
    parser.add_argument('--no-trim_to_gametime', dest='trim_to_gametime', action='store_false')
    parser.set_defaults(trim_to_gametime=False)


    args = parser.parse_args()
    main(args)
