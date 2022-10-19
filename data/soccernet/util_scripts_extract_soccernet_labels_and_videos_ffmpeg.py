import glob
import subprocess
import datetime
import os
import argparse
import cv2

# import moviepy
# from moviepy.editor import *

# def get_video_duration(video_file_path):
#     """
#     Get video duration in secs at video_file_path.

#     :param video_file_path: path to the file, e.g. ./abc/v_123.mp4.
#     :return: a float number for the duration.
#     """
#     get_duration_cmd = ('ffprobe -i "%s" -show_entries format=duration ' +
#                         '-v quiet -of csv="p=0"')
#     output = subprocess.check_output(
#         get_duration_cmd % video_file_path,
#         shell=True,  # Let this run in the shell
#         stderr=subprocess.STDOUT)
#     return float(output)

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

        cap = cv2.VideoCapture(filename)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print('total frames: {} \t fps: {} \t size: {},{}'.format(total, fps, h, w))

        # video = VideoFileClip(filename)

        duration = total / fps
        for start in range(0, int(duration), int(args.clip_length)):
            if start + args.clip_length > duration:
                break
            # if start == 0:
            #     effective_start = 0
            #     # 80 s
            # else:
            #     effective_start = start - 10

            start_time_str = str(datetime.timedelta(seconds=start))
            new_filename = os.path.join(args.output_folder, new_shortname_root).replace('.mkv', '.{}.{}.mp4'.format(start,args.clip_length))
            new_filename = new_filename.replace(" ", "")

            # command = f'ffmpeg -ss {start_time_str} -i "{filename}" \
            #     -c:v libx264 -c:a aac -strict experimental -b:a 98k \
            #     -t {str(datetime.timedelta(seconds=args.clip_length))} "{new_filename}"'

            command = f'ffmpeg -ss {start_time_str} -i "{filename}" \
                -t {str(datetime.timedelta(seconds=args.clip_length))} "{new_filename}"'

            os.system(command)

            # video_clip = video.subclip(start, start + args.clip_length)
            # video_clip.write_videofile(new_filename)


            # print(command)

        # last clip
        # effective_start = int(duration) - 80
        # start_time_str = str(datetime.timedelta(seconds=effective_start))
        # new_filename = os.path.join(args.output_folder, new_shortname_root).replace('.mkv', '.{}.{}.{}.mkv'.format(start_time_str.replace(':','-'),effective_start,args.clip_length))

        # command = f'ffmpeg -ss {start_time_str} -i "{filename}" \
        #     -c:v libx264 -c:a aac -strict experimental -b:a 98k \
        #     -t {str(datetime.timedelta(seconds=args.clip_length))} "{new_filename}"'
        # print(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default = '/home/zhangyuxuan07/PaddleSports/SoccerData/smallsetofdata')
    parser.add_argument('--output_folder', type=str, default = '/home/zhangyuxuan07/PaddleSports/SoccerData/output/video_clips')
    parser.add_argument('--clip_length', type=int, default = 10)

    args = parser.parse_args()
    main(args)


# add label for train, val, test

# add counter

# generate extraction script

# generate negative samples? or a collection of negative sample time-points?