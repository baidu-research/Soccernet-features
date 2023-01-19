import numpy as np
import os
from scipy.special import expit
from scipy.special import softmax
import glob
import json
from SoccerNet.Evaluation.ActionSpotting import evaluate
from tqdm import tqdm
import argparse

nms_window_size = 15

soccernet_path = '/mnt/data/zhiyu/SoccerNetv2_features/'
video_ini_root = '/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/'
k_labels_mapping_file = 'label_mapping.dense.txt'

def main(args):
    features_root = args.features_root
    result_jsons_root = args.result_jsons_root
    label_index_to_category_map = {}
    with open(k_labels_mapping_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            label_index_to_category_map[int(parts[0])] = parts[1]

    # Sample labels file /mnt/data/zhiyu/SoccerNetv2_features/spain_laliga/2016-2017/2017-05-21 - 21-00 Malaga 0 - 2 Real Madrid/Labels-v2.json

    def parse_gamestart_secs_line(line):
        # get video_ini
        # "/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/video.ini"
        # sample
        # [1_HQ.mkv]
        # start_time_second = 67

        # [2_HQ.mkv]
        # start_time_second = 52
        return int(line.split('=')[-1])

    def get_spot_from_NMS(Input, window=15, thresh=0.0):
        detections_tmp = np.copy(Input)
        indexes = []
        MaxValues = []
        while(np.max(detections_tmp) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(detections_tmp)
            max_index = np.argmax(detections_tmp)
            MaxValues.append(max_value)
            indexes.append(max_index)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
            detections_tmp[nms_from:nms_to] = -1
        
        return indexes, MaxValues

    def compute_nms():
        k_label_filename = 'Labels-v2.json'
        label_filenames_all = glob.glob(os.path.join(soccernet_path, f'**/{k_label_filename}'), recursive = True)

        for label_filename in tqdm(label_filenames_all):
            # video files here # /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data

            # 1st and 2nd half

            parts = label_filename.split('/')
            url_local = '/'.join(parts[-4:-1])
            features_shortname = '.'.join(parts[-4:-1]).replace(" ", "_")

            # get video_ini

            # remove the front of the features
            videos_starts_filename = label_filename.replace(k_label_filename, 'video.ini').replace(soccernet_path, video_ini_root)
            if not os.path.exists(videos_starts_filename):
                continue

            with open(videos_starts_filename, 'r') as g:
                lines = g.readlines()

            game_start_secs_in_videos = [parse_gamestart_secs_line(lines[1]), parse_gamestart_secs_line(lines[4])]

            # feature_folder needs to be changed
            detection_results_json = {}
            detection_results_json['UrlLocal'] = url_local
            predictions = []

            result_folder = label_filename.replace(k_label_filename, '').replace(soccernet_path, result_jsons_root)
            # print(result_folder)
            os.makedirs(result_folder, exist_ok=True)

            for half in [1, 2]:
                feature_folder = os.path.join(features_root, f'{features_shortname}.{half}_LQ').replace(" ", "_")
                features_filename = os.path.join(feature_folder, 'features.npy')

                # import ipdb; ipdb.set_trace()

                if not os.path.exists(features_filename):
                    print('missing feature', features_filename)
                    continue

                features = np.load(features_filename, allow_pickle = True)
                # import ipdb; ipdb.set_trace()
                backbone_features = features.item()['features']
                # print(features.item().keys())

                cls_score = expit(features.item()['cls_score_all'])
                event_times = features.item()['event_times']

                if len(cls_score.shape) == 3:
                    cls_score = np.squeeze(cls_score, axis = 1)
                    event_times = np.squeeze(event_times, axis = 1)
                    backbone_features = np.squeeze(backbone_features, axis = 1)

                cls_score = softmax(cls_score, axis = 1)
                # cls_score[1:] = softmax(cls_score[1:], axis = 1)

                # print(cls_score.shape, event_times.shape)
                # print(cls_score[0])
                # print(event_times[0])

                # import ipdb; ipdb.set_trace()

                original_fps = 1
                target_fps = 5

                # offset by gamestart
                half_index = half - 1
                feature_start = game_start_secs_in_videos[half_index] * original_fps
                feature_end = feature_start + 45 * 60
                backbone_features = backbone_features[feature_start: feature_end,:]
                cls_score = cls_score[feature_start: feature_end,:]
                event_times = event_times[feature_start: feature_end,:]

                features_trimmed_filename = os.path.join(feature_folder, 'features.trimmed.npy')
                with open(features_trimmed_filename, 'wb') as trimmed_file:
                    np.save(trimmed_file, {'features': backbone_features, 'cls_score': cls_score, 'event_times': event_times})

                probability_array = np.zeros((int(cls_score.shape[0] / original_fps) * target_fps, cls_score.shape[1])) - 1

                # print(probability_array.shape)
                # last one is background?

                for time_step in range(cls_score.shape[0]):
                    for category in range(cls_score.shape[1]):
                        # index_in_target = int(time_step * target_fps / original_fps + event_times[time_step, category])
                        index_in_target = int(time_step * target_fps / original_fps)
                        probability_array[index_in_target][category] = max(
                            cls_score[time_step, category], probability_array[index_in_target][category])

                features_array_filename = os.path.join(result_folder, f'{half}_features_array_{args.crop}.npy')
                with open(features_array_filename, 'wb') as f:
                    np.save(features_array_filename, backbone_features)
                    # print('wrote', features_array_filename)

                # print(probability_array[:50,])

            # need softmax on cls_score


                # indexes, MaxValues = get_spot_from_NMS(probability_array[:,0], window = int(15 * target_fps / original_fps))
                # print(len(indexes))
                # print(indexes[:10])
                # print(MaxValues[:10])

                # indexes, MaxValues = get_spot_from_NMS(probability_array[:,1], window = int(15 * target_fps / original_fps))
                # print(indexes[:10])
                # print(MaxValues[:10])

                for label_index in range(len(label_index_to_category_map)):
                    label = label_index_to_category_map[label_index]
                    if label == 'background':
                        continue

                    indexes, MaxValues = get_spot_from_NMS(probability_array[:,label_index], window = int(nms_window_size * target_fps / original_fps))

                    for i in range(len(indexes)):
                        time_step_index = indexes[i]
                        total_seconds = int(time_step_index  / (target_fps / original_fps))
                        minutes = int(total_seconds / 60)
                        seconds = total_seconds % 60

                        prediction = {
                            "gametime": f'{half} - {minutes}:{seconds}',
                            "label": label,
                            "position": f'{total_seconds * 1000}',
                            "half": f'{half}',
                            "confidence": MaxValues[i]
                        }

                        predictions.append(prediction)

            detection_results_json['predictions'] = predictions
            result_filename = os.path.join(result_folder, 'results_spotting.json')
            with open(result_filename, 'w') as f:
                json.dump(detection_results_json, f)

    compute_nms()

    results =  evaluate(SoccerNet_path=soccernet_path, 
                    Predictions_path=result_jsons_root,
                    split="valid",
                    prediction_file="results_spotting.json", 
                    version=2)

    print("Average mAP: ", results["a_mAP"])
    print("Average mAP per class: ", results["a_mAP_per_class"])
    print("Average mAP visible: ", results["a_mAP_visible"])
    print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
    print("Average mAP unshown: ", results["a_mAP_unshown"])
    print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])


# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Evaluation
# python EvaluateSpotting.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/


# evaluation:
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/20f2f74007c82b68a73c519dff852188df4a8b5a/Evaluation/EvaluateSpotting.py




# {
#     "UrlLocal": "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
#     "predictions": [ # list of predictions
#         {
#             "gameTime": "1 - 0:31", # format: "{half} - {minutes}:{seconds}",
#             "label": "Ball out of play", # label for the spotting,
#             "position": "31500", # time in milliseconds,
#             "half": "1", # half of the game
#             "confidence": "0.006630070507526398", # confidence score for the spotting,
#         },
#         {
#             "gameTime": "1 - 0:39",
#             "label": "Foul",
#             "position": "39500",
#             "half": "1",
#             "confidence": "0.07358131557703018"
#         },
#         {
#             "gameTime": "1 - 0:55",
#             "label": "Foul",
#             "position": "55500",
#             "half": "1",
#             "confidence": "0.20939764380455017"
#         },
#         ...
#     ]
# }

# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/20f2f74007c82b68a73c519dff852188df4a8b5a/Task1-ActionSpotting/TemporallyAwarePooling/src/train.py

# we need to modify to change to include offset as well, times and probabilities (will be slow)?
                # def get_spot_from_NMS(Input, window=60, thresh=0.0):

                #     detections_tmp = np.copy(Input)
                #     indexes = []
                #     MaxValues = []
                #     while(np.max(detections_tmp) >= thresh):

                #         # Get the max remaining index and value
                #         max_value = np.max(detections_tmp)
                #         max_index = np.argmax(detections_tmp)
                #         MaxValues.append(max_value)
                #         indexes.append(max_index)
                #         # detections_NMS[max_index,i] = max_value

                #         nms_from = int(np.maximum(-(window/2)+max_index,0))
                #         nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                #         detections_tmp[nms_from:nms_to] = -1



# results seem wrong

# def apply_nms(series, window = 60, )

#     results =  evaluate(SoccerNet_path=dataloader.dataset.path, 
#                  Predictions_path=output_results,
#                  split="test",
#                  prediction_file="results_spotting.json", 
#                  version=2)

# print("Average mAP: ", results["a_mAP"])
# print("Average mAP per class: ", results["a_mAP_per_class"])
# print("Average mAP visible: ", results["a_mAP_visible"])
# print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
# print("Average mAP unshown: ", results["a_mAP_unshown"])
# print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', type=str, default = '/mnt/storage/gait-0/xin/soccernet_features_5_ppTimeSformer_balanced/')
    parser.add_argument('--result_jsons_root', type=str, default = '/mnt/storage/gait-0/xin/soccernet_features_result_jsons/',
        help = 'This is where fature files will be written')
    parser.add_argument('--crop', type=str, default = '')

# features_root = '/mnt/storage/gait-0/xin/soccernet_features_5_ppTimeSformer_balanced/'
# result_jsons_root = '/mnt/storage/gait-0/xin/soccernet_features_result_jsons/'

    args = parser.parse_args()
    main(args)