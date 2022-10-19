import os
filename = '/mnt/storage/gait-0/xin/dataset/soccernet_456x256/annotation.txt'

script_file = '/mnt/storage/gait-0/xin/dev/PaddleVideo/data/soccernet/generate_training_short_clips.sh'

commands = {}
with open(script_file, 'r') as f:
    script_lines = f.readlines()
    for script_line in script_lines:
        key = script_line.split()[-1].replace('"','')  
        commands[key] = script_line

with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        video_file = line.split()[0]
        if not os.path.exists(video_file):
            # print(video_file)
            if video_file in commands:
                command = commands[video_file]
                new_command = command.replace('-vf scale=456x256', '-vf scale=456x256 -map 0:v -map 0:a -c copy ')
                print(new_command)
            else:
                print(f'# no command for {video_file}')

# europe_uefa-champions-league.2015-2016.2015-09-15_-_21-45_Galatasaray_0_-_2_Atl._Madrid.1_HQ.0-51-00.3070.10.mkv