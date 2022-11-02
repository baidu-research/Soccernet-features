![Soccernet Event Detection](image.gif)

# Preface

This repo contains code to finetune backbone models on the Soccernet dataset. The Soccernet features are used in down-stream tasks, in particular event spotting and replay grounding. In 2021 and 2022, the winning solutions for the Soccernet Challenge localization track used finetuned features generated this way. In the next section, links to the previous best features are attached. This repo is made public so that further progress on feature pretraining can be made or inference could be done on data other than Soccernet (If you do either of these things, do not forget to give a reference to this work). Once better features are generated, this repo will be updated.

## Best Pretrained Soccernet Features from 2021

This other [repo](https://github.com/baidu-research/vidpress-sports) contains the pretrained Soccernet features winning the CVPR 2021 ActivityNet Challange, Temporal Localization track, SoccerNet Challenge for 2021 and 2022. The features were extracted from an ensemble of 5 models. Those are the best features known for Soccernet so far. 

# Train your own model

## Generate low resolution clips

In this section, we will be extracting short 10 seconds clips from the Soccernet videos and the clips will have a lower resolution for training.

Download raw HQ video data from the [Soccernet official website](https://www.soccer-net.org/download). Put it in a folder and set $RAW_VIDEOS_ROOT to that folder.

Run the following command. The output is commands to extract the clips. Redirect the output into a file because there are many extraction commands and you will need to split the file to run the commands in parallel. Choose a folder and set the environment variable $CLIPS_FOLDER to save your clips.

    python data/soccernet/generate_training_short_clips.py --input_folder $RAW_VIDEOS_ROOT --clips_folder=$CLIPS_FOLDER > data/soccernet/generate_training_short_clips.sh

Make a folder to save the paralle scripts:

    mkdir data/soccernet/short_clips_parallel

This is a sample to split into 400 equal parts:

    for i in {0..399};
    do
        sed -n ${i}~400p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel/${i}.sh;
    done

The commands may get stuck on a few videos and render the jobs stuck, so here is another split into 401 parts to run after the above job.

    for i in {0..400};
    do
        sed -n ${i}~401p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel_401/${i}.sh;
    done

## Run the jobs

Sample code to run them on a slurm based cluster:

    for i in {0..399};
    do
    sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet/short_clips_parallel/${i}.sh" \
    --output="data/soccernet/short_clips_parallel/${i}.log"
    done

For the seconds split,

    for i in {0..400};
    do
    sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet/short_clips_parallel_401/${i}.sh" \
    --output="data/soccernet/short_clips_parallel_401/${i}.log"
    done

## Generate label files

Set $RAW_VIDEOS_ROOT to the root of the folder where all HQ Soccernet videos are downloaded into. Set $LABELS_ROOT to the folder where the labels are.

    python data/soccernet/generate_labels.py \
    --extension mkv \
    --raw_videos_root $RAW_VIDEOS_ROOT \
    --labels_root $LABELS_ROOT \
    --clips_folder $CLIPS_FOLDER

Convert the labels files into one list annotation.txt

    python data/soccernet/labels_to_pdvideo_format.py \
    --clips_folder $CLIPS_FOLDER \
    --output_folder $CLIPS_FOLDER

Split into train val test. $SPLITS_FOLDER contains the Soccernet numpy files which contain the train, val, test match names.

    python data/soccernet/split_annotation_into_train_val_test.py \
    --splits_folder $SPLITS_FOLDER \
    --annotation_file $CLIPS_FOLDER/annotation.txt \
    --clips_folder $CLIPS_FOLDER

Now you should have train.list, val.list, test.list in $CLIPS_FOLDER that are in kinetics format that can be trained with any video classification model.

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

Make paddle label files?

# Inference on whole video files

## Convert video input into lower resolution

This generates a sample script that converts all of the Soccernet videos.

    python data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py \
    --input_folder /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference > \
    data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh

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

## Generate json labels

    python data/soccernet_dense_anchors/generate_whole_video_inference_jsons.py \
    --videos_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists

# Train command

    python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams

# Inference command

    python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_videoswin_test  main.py  --test -c data/soccernet/soccernet_videoswin_k400_dense_one_file_inference.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams
