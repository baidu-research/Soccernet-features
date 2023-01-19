from collections import defaultdict
import math
import json

def balance_list(filename, output_file):
    # filename = 'train.dense.list'
    # output_file = 'train.dense.balanced.list'

    label_counters = defaultdict(int)

    def parse_label(line):
        line_data = json.loads(line.strip())
        label = line_data['label']
        return label

    def parse_match(line):
        line_data = json.loads(line.strip())
        filename = line_data['filename']
        match = filename.split('_HQ')[0]
        return match

    def approximate_timestamp(line):
        line_data = json.loads(line.strip())
        filename = line_data['filename']
        timestamp = int(filename.split('.')[-3])
        return timestamp

    # some label filtering since some match videos contain a lot of content before or after 
    match_started = False
    current_match = None

    match_label_starts_ends = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line_data = json.loads(line.strip())
            match = parse_match(line)
            label = parse_label(line)
            timestamp = approximate_timestamp(line)

            if match not in match_label_starts_ends:
                match_label_starts_ends[match] = {'start': 1e6, 'end': -1}

            if not label == 17:
                match_label_starts_ends[match]['start'] = min(match_label_starts_ends[match]['start'], timestamp)
                match_label_starts_ends[match]['end'] = max(match_label_starts_ends[match]['end'], timestamp)
                match_label_starts_ends[match]['end'] = max(match_label_starts_ends[match]['end'], match_label_starts_ends[match]['start'] + 45 * 60)

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line_data = json.loads(line.strip())
            match = parse_match(line)
            label = parse_label(line)
            timestamp = approximate_timestamp(line)

            start = match_label_starts_ends[match]['start']
            end = match_label_starts_ends[match]['end']

            if timestamp >= start and timestamp <= end:
                label_counters[label] += 1

    print('label_counters')
    print(label_counters)

    label_max = max(label_counters.values())

    label_intended = 21000

    repeats = {}
    for key in label_counters:
        repeats[key] = math.ceil(label_intended / label_counters[key])

    print('repeats')
    print(repeats)

    with open(output_file, 'w') as f:
        for line in lines:
            line_data = json.loads(line.strip())
            match = parse_match(line)
            label = parse_label(line)
            timestamp = approximate_timestamp(line)

            if timestamp >= start and timestamp <= end:
                for _ in range(repeats[label]):
                    f.write(line)
    print('wrote', output_file)

balance_list('/mnt/storage/gait-0/xin/dev/PaddleVideo/val.dense.list', '/mnt/storage/gait-0/xin/dev/PaddleVideo/val.dense.balanced.list')
