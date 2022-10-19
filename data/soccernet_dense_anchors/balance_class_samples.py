from collections import defaultdict
import math
import json

filename = 'train.dense.list'
label_counters = defaultdict(int)

def parse_label(line):
    line_data = json.loads(line.strip())
    label = line_data['label']
    return label

with open(filename, 'r') as f:
    lines = f.readlines()

    for line in lines:
        label = parse_label(line)
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

output_file = 'train.dense.balanced.list'
with open(output_file, 'w') as f:
    for line in lines:
        label = parse_label(line)
        for _ in range(repeats[label]):
            f.write(line)

