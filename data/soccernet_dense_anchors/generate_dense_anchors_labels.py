import json
import glob
import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

k_background_label = 17

def main(args):
    labels_folder = args.clips_folder
    output_folder = args.output_folder
    label_file = os.path.join(output_folder, 'label_mapping.dense.txt')
    annotation_file = os.path.join(output_folder, 'dense.list')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # files = sorted(glob.glob(os.path.join(labels_folder, '*.json'), recursive= True))
    with open(os.path.join(labels_folder, 'json_list.txt'), 'r') as f:
        lines = f.readlines()
    files = [line.strip() for line in lines]

    labels = []
    num_labels_list = []
    for filename in tqdm(files):
        # print(filename)
        with open(filename, 'r') as f:
            data = json.load(f)
            num_labels = len(data['annotations'])
            num_labels_list.append(num_labels)
            for annotation in data["annotations"]:
                labels.append(annotation["label"])

    print('len(num_labels_list)', len(num_labels_list))
    counter = Counter(num_labels_list)
    print('num_labels_list counter', counter)

    labels = list(dict.fromkeys(labels))
    labels.append('background')
    print('number of labels', len(labels))

    dict_labels = {labels[i] : i for i in range(len(labels))}

    # this file contains the text to label index mapping
    with open(label_file, "w") as label_file:
        for i, label in enumerate(labels):
            label_file.write('{} {}\n'.format(i, label))

    def generate_list_of_dict(data):
        list_of_dict = []
        try:
            if len(data['annotations']) == 0:
                label_dict = {}
                #  {"filename":"/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_HQ.0-00-00.0.10.mkv", "label":0, "event_time":2.0, "clip_length_secs": 10.0}
                label_dict['filename'] = data['path']
                label_dict['label'] = k_background_label
                label_dict['event_time'] = 5.0
                label_dict['clip_length_secs'] = 10.0
                list_of_dict.append(label_dict)
            else:
                for i in range(len(data['annotations'])):
                    label_dict = {}
                    #  {"filename":"/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_HQ.0-00-00.0.10.mkv", "label":0, "event_time":2.0, "clip_length_secs": 10.0}
                    label_dict['filename'] = data['path']
                    label_dict['label'] = dict_labels[data["annotations"][i]["label"]]
                    label_dict['event_time'] = data["annotations"][i]["event_time"]
                    label_dict['clip_length_secs'] = data['clip_length']

                    list_of_dict.append(label_dict)
                    print('recorded {} actions in clip'.format(len(data['annotations'])))
        except Exception:
            print('0 actions in clip')
        return list_of_dict

    print('writing annotations file...')
    # with open(annotation_file_dir, "w") as annotation_file:
    list_of_dict_all = []
    for filename in tqdm(files):
        with open(filename, "r") as f:
            data = json.load(f)
            list_of_dict_all += generate_list_of_dict(data)
    with open(annotation_file, "w") as annotation_file:
        for item in list_of_dict_all:
            annotation_file.write(json.dumps(item) + '\n')
            
                # if len(data['annotations']) == 1:
                #     annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][0]["label"]]))
                # elif len(data['annotations']) == 0:
                #     annotation_file.write('{} {}\n'.format(data['path'], dict_labels['background']))
                # elif len(data['annotations']) >= 2:
                #     for i in range(len(data['annotations'])):
                #         annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][i]["label"]]))
                    # annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][1]["label"]]))
                # else:
                #     print('warning: {} has more than 2 labels'.format(data['path']), data['annotations'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_folder', type=str, required = True, help = 'Where json annotation files are')
    parser.add_argument('--output_folder', type=str, required = True, help = 'Where label_mapping.dense.txt and dense.list are saved')

    args = parser.parse_args()
    main(args)