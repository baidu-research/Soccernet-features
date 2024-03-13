![Soccernet Event Detection](image.gif)

# Preface

This repo contains code to run feature inference and finetune backbone models on the [Soccernet](https://www.soccer-net.org/home) dataset. The Soccernet features are used in down-stream tasks, in particular event spotting and replay grounding. The winning solutions for the CVPR 2021 and 2022 ActivityNet Challange, Temporal Localization track, used features from this [Soccernet Baidu Features repo](https://github.com/baidu-research/vidpress-sports). This repo makes a single model based on TimeSFormer fully opensource with pretrained weights available to run inference or train on any video. 

# Understanding the result features

Load the features in as follows:

    import numpy as np
    a = np.load('features.py')

The whole numpy array could look like this:

    array({'cls_score_all': array([[[-1.1469777,  1.2446018,  1.7308334, ..., -3.7974577,
             -3.9211693,  9.876837 ]],

       [[-1.1502348,  1.2446928,  1.7270398, ..., -3.7986917,
         -3.922049 ,  9.873158 ]],

       [[-1.1553113,  1.2458539,  1.7238224, ..., -3.8000093,
         -3.9231768,  9.866276 ]],

       ...,

       [[-1.1867781,  1.2394937,  1.7207626, ..., -3.7993345,
         -3.9266846,  9.682111 ]],

       [[-1.1717311,  1.3373474,  1.7501842, ..., -3.809371 ,
         -3.9297879,  9.226656 ]],

       [[-1.0841012,  1.4193138,  1.7844383, ..., -3.8307285,
         -3.9134252,  8.695862 ]]], dtype=float32), 'event_times': array([[[0.61532754, 0.48600802, 0.384552  , ..., 0.4478651 ,
         0.63936603, 1.        ]],

       [[0.595665  , 0.4832232 , 0.37086746, ..., 0.45608336,
         0.63960594, 1.        ]],

       [[0.57046086, 0.48085424, 0.35832942, ..., 0.46679202,
         0.63987064, 1.        ]],

       ...,

       [[0.5881998 , 0.47877887, 0.51381993, ..., 0.42987552,
         0.64069575, 1.        ]],

       [[0.52614266, 0.48890987, 0.49975097, ..., 0.43349805,
         0.63012475, 1.        ]],

       [[0.61633825, 0.48636538, 0.46834537, ..., 0.5310676 ,
         0.6127171 , 1.        ]]], dtype=float32), 'features': array([[[-0.0132505 ,  0.02137352, -0.21291943, ..., -0.07798928,
          0.21858051, -0.23259796]],

       [[-0.01204887,  0.02450565, -0.21337315, ..., -0.07861489,
          0.21864586, -0.23138241]],

       [[-0.01019135,  0.02715371, -0.21588442, ..., -0.07936225,
          0.2186201 , -0.22969423]],

       ...,

       [[ 0.01318725, -0.02435086, -0.12325487, ..., -0.00325756,
          0.21175556, -0.11824775]],

       [[-0.00307255, -0.02015326, -0.10546789, ...,  0.02826597,
          0.18887816, -0.095611  ]],

       [[-0.01093192, -0.06524777, -0.15625732, ...,  0.0292125 ,
          0.17296971, -0.11872528]]], dtype=float32)}, dtype=object)

Extract the useful features (other ones were left unimplemented):

    data_dict = a.item() 
    features = data_dict['features']

features.shape = (3489, 1, 768). The old soccernet features have a shape of (number of frames, 8576) being an ensemble of models. In the new feature, the feature dimension is reduced to just 768, 3489 is the number of frames. One still need to train a stage 2 model to use these features (I do not have access to the one I verified the features with any more. Let me know if someone trained one and wants to share. )

# Feature inference

This pipeline can be used for extracting features for Soccernet videos or other broadcast soccer videos in general.

Install cuda、cudnn、nccl first. Clone this repo.

    pip3 install paddlepaddle-gpu --upgrade

    pip3 install --upgrade -r requirements.txt

A config file for the video file in the follow format needs to be constructed and its filename set as VIDEO_CONFIG. It needs to have the path to the video file, fps of the video and the total length in seconds.

    {
        "filename": "/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_5fps/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_LQ.mkv",
        "fps": 5,
        "length_secs": 3196
    }

Download weight file and set the filename to be WEIGHT_FILE. 
    
- [Microsoft OneDrive](https://1drv.ms/u/s!AruitsssaVf8edpJuWH1a1KCLCc?e=xkgpdn) 
- [Baidu Pan](https://pan.baidu.com/s/10b5pdwwxNXezWg6-_z3YwA?pwd=segt)

Set enviroment variables LOG_DIR to save logs. The test config do not need to be changed. Set INFERENCE_DIR. This is where the features.npy containing the weights will be saved.

    python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=$LOG_DIR  main.py  \
    --test -c data/soccernet_inference/soccernet_pptimesformer_k400_one_file_inference.yaml \
    -w $WEIGHT_FILE -o inference_dir=$INFERENCE_DIR \
    -o DATASET.test.file_path=$VIDEO_CONFIG

Inference on a 45 min video at 5fps takes approximately 2-3 hours.

# Train your own model

## Generate low resolution clips

In this section, we will be extracting short 10 seconds clips from the Soccernet videos and the clips will have a lower resolution for training.

Download raw HQ video data from the [Soccernet official website]([https://www.soccer-net.org/download](https://www.soccer-net.org/home)). Put it in a folder and set $RAW_VIDEOS_ROOT to that folder.

Notes:

- Three resolutions of videos are provided 1080p, 720p, 224p. The caveat here is that the scripts assumes 1080p is used. The key difference is that compared to the 720p, 224p videos, which are trippmed from the gamestart to the end of the game. Therefore, a lot of the offsets in the scripts need to be removed.

- The code as is only lowers the resolution for training clips. Potential resource saving (decoding time and memory) can be achieved using even lower fps. Decoding speed is the bottleneck in training. Lower fps sacrifices data diversity a little bit. This tradeoff was not investigated but is worth considering if resource constrained.

Run the following command. The output is commands to extract the clips. Redirect the output into a file because there are many extraction commands and you will need to split the file to run the commands in parallel. Choose a folder and set the environment variable $CLIPS_FOLDER to save your clips.

    python data/soccernet/generate_training_short_clips.py --input_folder $RAW_VIDEOS_ROOT --clips_folder=$CLIPS_FOLDER > data/soccernet/generate_training_short_clips.sh

Make a folder to save the paralle scripts:

    mkdir data/soccernet/short_clips_parallel

This is a sample to split into 400 equal parts:

    for i in {0..399};
    do
        sed -n ${i}~400p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel/${i}.sh;
    done

Manchester_United_1_-_1_Liverpool.1_HQ

The commands may get stuck on a few videos and render the jobs stuck, so here is another split into 401 parts to run after the above job.

    for i in {0..400};
    do
        sed -n ${i}~401p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel_401/${i}.sh;
    done

## Generate label files

Set $RAW_VIDEOS_ROOT to the root of the folder where all HQ Soccernet videos are downloaded into. Set $LABELS_ROOT to the folder where the labels are.

    python data/soccernet/generate_labels.py \
    --extension mkv \
    --raw_videos_root $RAW_VIDEOS_ROOT \
    --labels_root $LABELS_ROOT \
    --clips_folder $CLIPS_FOLDER

## Run the jobs

Sample code to run them on a slurm based cluster:

    for i in {0..399};
    do
    sbatch -p 1080Ti,2080Ti  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet/short_clips_parallel/${i}.sh" \
    --output="data/soccernet/short_clips_parallel/${i}.log"
    done

For the seconds split,

    for i in {0..400};
    do
    sbatch -p 1080Ti,2080Ti,M40x8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet/short_clips_parallel_401/${i}.sh" \
    --output="data/soccernet/short_clips_parallel_401/${i}.log"
    done

# Prepare training with class and event time labels

Generate label_mapping.txt (for category to category index map) and dense.list files.

    python data/soccernet_dense_anchors/generate_dense_anchors_labels.py \
    --clips_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256 \
    --output_folder ./

Split into train, val, test

    python data/soccernet/split_annotation_into_train_val_test.py \
    --annotation_file dense.list \
    --clips_folder ./ \
    --mode json

Then you will have train.list, val.list, test.list in this folder.

## Training command

    python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams

## Training slurm command

    sbatch -p V100x8_mlong --gres=gpu:8 --cpus-per-task 40 -n 1  \
    --wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
    --output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense.log"

## Training log link

The feature extractor training log is here. The folder also contains the trained NetVLAD++ action detection result (final mAP by category).
Soccernet-features/data/soccernet/archived_logs/

# Inference on whole video files

Could be convenient to consider gamestart.

## Convert video input into lower resolution

This generates a sample script that converts all of the Soccernet videos. We have to be a little bit careful with if the video starts from gamestart or not. (trim_to_gametime)

    python data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py \
    --input_folder /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference > \
    data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh

## Converting video input into 5fps

In order to run inference on half game videos, besides lower resolution to 456x256, the videos had to be run on a lower fps. (The inference sampler decodes the whole video. Then it extracts clips of time intervals [0,n), [1,n+1) etc.

## Parallelize resolution conversion

Each 45 min video files takes about 10 min to convert to lower resolution. So we parallelize to 100 such jobs.

    for i in {0..99};
    do
    sed -n ${i}~100p data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh > data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh;
    done

Run the parallel jobs on a cluster, slurm based for example.

    for i in {0..99};
    do
    sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh" \
    --output="data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.log"
    done

## Generate json videos list for inference to read

    python data/soccernet_dense_anchors/generate_whole_video_inference_jsons.py \
    --videos_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists

## Sample inference command

    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00007.pdparams
    INFERENCE_JSON_CONFIG=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ.mkv
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features
    SHORTNAME=`basename "$INFERENCE_JSON_CONFIG" .mkv`
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$SHORTNAME
    echo $INFERENCE_DIR

    mkdir $INFERENCE_DIR

    python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_videoswin_test  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG


## Run all inference 5fps

Needed to sample down to 5fps because of our scale to run all Soccernet data

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100_balanced/ppTimeSformer_dense_event_lr_100_fc_lr_100_balanced_epoch_00004.pdparams

INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_4_ppTimeSformer
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_5_ppTimeSformer_balanced


for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
do 
line=`basename "$FILE" .mkv`
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

echo "rm /mnt/storage/gait-0/xin//logs/$line.log"
echo "sbatch -p 1080Ti --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
--output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
done > unfinished_inference.sh

## Check inference logs

These command help you check the last lines of the jobs and see if any bug arose.

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features

for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*;
do
line=`basename "$FILE" .mkv`
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

log_file=/mnt/storage/gait-0/xin//logs/$line.log
echo $log_file
tail -n 1 /mnt/storage/gait-0/xin//logs/$line.log
tail -n 1 /mnt/storage/gait-0/xin//logs/$line/workerlog.0
done

## Find unfinished jobs
python data/soccernet_dense_anchors/check_unfinished_inference.py \
--inference_root /mnt/storage/gait-0/xin/soccernet_features_4_ppTimeSformer/ > inference_matches_todo.txt

## Rerun unfinished jobs 5fps

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features


cat inference_matches_todo.txt | while read line
do
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

echo "sbatch -p 1080Ti --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
--output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
done > unfinished_inference.sh



# Rerun unfinished full fps

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features


cat inference_matches_todo.txt | while read line
do
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

echo "sbatch -p 1080Ti,2080Ti,TitanXx8 --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
--output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
done

## Inspect logs
Errors:

    /mnt/storage/gait-0/xin//logs/spain_laliga.2015-2016.2015-09-12_-_17-00_Espanyol_0_-_6_Real_Madrid.1_LQ.log

Runtime:

    /mnt/storage/gait-0/xin//logs/spain_laliga.2015-2016.2015-09-12_-_17-00_Espanyol_0_-_6_Real_Madrid.1_LQ/workerlog.0


Note:

nano /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps//england_epl.2016-2017.2016-11-06_-_19-30_Leicester_1_-_2_West_Brom.1_LQ.mkv
override time to 3780 secs
