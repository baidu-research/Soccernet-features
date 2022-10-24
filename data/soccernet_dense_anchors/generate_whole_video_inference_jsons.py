import glob
import os
import subprocess
import argparse
import json

def get_video_duration(video_file_path):
    """
    Get video duration in secs at video_file_path.
    :param video_file_path: path to the file, e.g. ./abc/v_123.mp4.
    :return: a float number for the duration.
    """
    get_duration_cmd = ('ffprobe -i "%s" -show_entries format=duration ' +
                        '-v quiet -of csv="p=0"')
    output = subprocess.check_output(
        get_duration_cmd % video_file_path,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT)
    return float(output)

def main(args):
    # sample filename /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2016-2017/2016-08-20 - 19-15 Barcelona 6 - 2 Betis/1_HQ.mkv
    files = sorted(glob.glob(os.path.join(args.videos_folder, f'*_LQ.{args.extension}')))
 
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for filename in files:
        try:
            # print('processing', filename)
            duration = get_video_duration(filename)

            # {
            #     "filename": "/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2014-2015.2015-05-17_-_18-00_Manchester_United_1_-_1_Arsenal.1_HQ.0-00-00.0.10.mkv",
            #     "fps": 25,
            #     "length_secs": 10
            # }
            data = {
                'filename': filename,
                'fps': args.fps,
                'length_secs': int(duration)
            }

            json_filename = filename.replace(args.videos_folder, args.output_folder)
            with open(json_filename, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(filename)
            # print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_folder', type=str, default = '/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference')
    parser.add_argument('--output_folder', type=str, default = '/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists')
    parser.add_argument('--fps', type=int, default = 25)
    parser.add_argument('--extension', type=str, default = 'mkv')


    args = parser.parse_args()
    main(args)
