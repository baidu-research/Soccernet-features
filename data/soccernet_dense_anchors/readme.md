# Copy source files and config files
    
    rsync -abviuzP paddlevideo/ $PADDLEVIDEO_SOURCE_FOLDER/

    rsync config or data files

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

generates files like train.dense.list in dense mode.

# Inference on whole video files

## Convert video input into lower resolution

This generates a sample script that converts all of the Soccernet videos.

    python data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py \
    --input_folder /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference > \
    data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh

Need to sample down for inference on whole matches
    python data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py \
    --input_folder /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_5fps \
    --fps 5 > \
    data/soccernet_inference/convert_video_to_lower_resolution_for_inference_5fps.sh

## Parallelize resolution conversion

Each 45 min video files takes about 10 min to convert to lower resolution. So we parallelize to 100 such jobs.

    for i in {0..99};
    do
    sed -n ${i}~100p data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh > data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh;
    done

    for i in {0..199};
    do
    sed -n ${i}~100p data/soccernet_inference/convert_video_to_lower_resolution_for_inference_5fps.sh > data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh;
    done

Run the parallel jobs on a cluster, slurm based for example.

    for i in {0..199};
    do
    sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh" \
    --output="data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.log"
    done

Check job status

    for i in {0..199};
    do
    echo $i
    cat data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.log | tail -n 1
    done

# Train command

    python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams

## Generate inference json job files

    python data/soccernet_dense_anchors/generate_whole_video_inference_jsons.py \
    --videos_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists


    python data/soccernet_dense_anchors/generate_whole_video_inference_jsons.py \
    --fps 5 \
    --videos_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_5fps \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps

## Sample inference command

    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00007.pdparams
    INFERENCE_JSON_CONFIG=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ.mkv
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features
    SHORTNAME=`basename "$INFERENCE_JSON_CONFIG" .mkv`
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$SHORTNAME
    echo $INFERENCE_DIR

    mkdir $INFERENCE_DIR

    python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_videoswin_test  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG 

## Run all inference 

Needed to sample down to 5fps because of our scale to run all Soccernet data

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features

for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*;
do 
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line


echo "sbatch -p 1080Ti --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
--output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
done

## Find unfinished jobs
python data/soccernet_dense_anchors/check_unfinished_inference.py \
--inference_root /mnt/storage/gait-0/xin/soccernet_features

# List of changed files and corresponding changes.

- Label files processing are changed and labels of category and event_times are composed into dicts to send into the pipeline. Class names are added into the init.
    
        paddlevideo/loader/dataset/video_dense_anchors.py

        paddlevideo/loader/dataset/__init__.py

    Added temporal coordinate embedding to inputs. Removed event time loss for background class. Added parser for one video file list.

        paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py

- Added EventSampler

        paddlevideo/loader/pipelines/sample.py

        paddlevideo/loader/pipelines/__init__.py

    Added sampling one whole video file.

        paddlevideo/loader/pipelines/sample_one_file.py
    
    Added decoder for just one file 

        paddlevideo/loader/pipelines/decode.py

- Multitask losses.

        paddlevideo/modeling/losses/dense_anchor_loss.py
        
        paddlevideo/modeling/losses/__init__.py

- Changed head output. Class and event times.

        paddlevideo/modeling/heads/i3d_anchor_head.py

        paddlevideo/modeling/heads/pptimesformer_anchor_head.py

        paddlevideo/modeling/heads/__init__.py

- Input and output format in train_step, val step etc.

        paddlevideo/modeling/framework/recognizers/recognizer_transformer_features_inference.py

        paddlevideo/modeling/framework/recognizers/recognizer_transformer_dense_anchors.py
        
        paddlevideo/modeling/framework/recognizers/__init__.py

- Add a new mode to log both class loss and event time loss.

        paddlevideo/utils/record.py

- Added MODEL.head.name and MODEL.head.output_mode branch to process outputs of class scores and event_times. Also unified feature inference with simple classification mode.

        paddlevideo/tasks/test.py

- Lower generate lower resolution script.

        data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py

- Balanced samples do not seem necessary 
    
        data/soccernet_dense_anchors/balance_class_samples.py

- Collate file to replace the current library file

        /mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/lib/python3.7/site-packages/paddle/fluid/dataloader/collate.py

- Config files

        data/soccernet/soccernet_videoswin_k400_dense_one_file_inference.yaml

- Updated to support dense anchors

        data/soccernet/split_annotation_into_train_val_test.py

# Comments

1. TODO paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py can inherit from paddlevideo/loader/dataset/video_dense_anchors.py





paddlevideo/loader/pipelines/decode.py




/mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/bin/python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/dense_anchors main.py --validate -c data/soccernet/soccernet_videoswin_k400_dense.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams


python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/dense_anchors main.py --validate -c data/soccernet/soccernet_videoswin_k400_dense.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams



python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_1 main.py --validate -c data/soccernet/soccernet_videoswin_k400_dense.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams 2>&1 | tee -a logs/dense_anchors_1.log

sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_1 main.py --validate -c data/soccernet/soccernet_videoswin_k400_dense.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_21_dense_lr_0.001.log"

sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_2 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.01.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_20_dense_lr_0.01.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_2_lr_0.001 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_21_dense_adamW_lr_0.001.log"

sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_2_lr_0.0001 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.0001.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_21_dense_adamW_lr_0.0001.log"

sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_no_warmup main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_no_warmup.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_no_warmup.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "/mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/bin/python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_balanced main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_balanced.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_balanced.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "/mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/bin/python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_20_dense_lr_0.00001_adamW main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.00001.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_20_dense_lr_0.00001_adamW.log"



sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "/mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/bin/python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_20_dense_lr_0.000001_adamW main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.000001.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_20_dense_lr_0.000001_adamW.log"


sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_2 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.00001.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_20_dense_lr_0.00001.log"


sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/dense_anchors_lr_0.1 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.1.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_21_dense_lr_0.1.log"


sbatch -p V100x8_mlong --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_randomization main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_pptimesformer_randomization.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_randomization.log"

sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale.log"

sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale.log"


sbatch -p V100x8_mlong --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale.log"



sbatch -p V100x8_mlong --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_lr_1e-4 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_lr_1e-4.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_lr_1e-4.log"


sbatch -p V100x8 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_lr_1e-5 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_lr_1e-5.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_lr_1e-5.log"


sbatch -p V100x8_mlong  --exclude asimov-231 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr_50_warmup main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr_50_warmup.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr_50_warmup.log"



sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_event_lr.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100.log"


sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_50 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_50.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_50.log"



python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_randomization main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_pptimesformer_randomization.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams

some augmentation error

在add_coordinates_embedding_to_imgs的时候pyav得到的是tensor， decord是np array? pyav decode完就是paddle.tensor了？

'decord'
ipdb> type(imgs)
<class 'numpy.ndarray'>
ipdb> imgs.shape
(3, 16, 256, 456)


python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense main.py --validate -c data/soccernet/soccernet_pptimesformer_k400_videos_dense.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams

python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense main.py --validate -c data/soccernet/soccernet_videoswin_k400_dense.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams



TODO:
Test one video inference, test on longer video


git filter-branch --index-filter \
    'git rm -rf --cached --ignore-unmatch data/soccernet/generate_training_short_clips.sh' HEAD

ffmpeg -i "/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford/1_HQ.mkv" -vf scale=456x256 -map 0:v -avoid_negative_ts make_zero -c:v libx264 -c:a aac "/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference/england_epl.2015-2016.2015-08-29_-_17-00_Manchester_City_2_-_0_Watford.1_LQ.mkv" -max_muxing_queue_size 9999



for i in {0..28};
do
sed -n ${i}~29p data/soccernet_inference/convert_video_rerun.sh > data/soccernet_inference/convert_video_rerun_parallel/${i}.sh;
done



for i in {0..28};
do
sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
"echo yes | bash data/soccernet_inference/convert_video_rerun_parallel/${i}.sh" \
--output="data/soccernet_inference/convert_video_rerun_parallel/${i}.log"
done



override weight, override jobs file

sbatch each line

weight file is always in the output folder. output/$model_name

for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/*; do echo $FILE; done

INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00007.pdparams
INFERENCE_JSON_CONFIG=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ.mkv
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features
SHORTNAME=`basename "$INFERENCE_JSON_CONFIG" .mkv`
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$SHORTNAME
echo $INFERENCE_DIR

mkdir $INFERENCE_DIR

python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_videoswin_test  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG 


for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/*; 


INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00012.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features


for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/*; 
do 
echo $FILE;
INFERENCE_JSON_CONFIG=$FILE
SHORTNAME=`basename "$INFERENCE_JSON_CONFIG" .mkv`
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$SHORTNAME
mkdir $INFERENCE_DIR

sbatch -p 1080Ti,2080Ti --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap "python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$SHORTNAME  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG" \
--output="/mnt/storage/gait-0/xin//logs/$SHORTNAME.log"

echo /mnt/storage/gait-0/xin//logs/$SHORTNAME.log

done



python -u -B -m paddle.distributed.launch --gpus="0" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_50_compare main.py --validate -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_compare.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams

Timesformer does not have coordinate embedding



/mnt/storage/gait-0/xin/soccernet_features/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ/features.npy

a = np.load("/mnt/storage/gait-0/xin/soccernet_features/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ/features.npy", allow_pickle = True)

a
array({'features': array([[[-0.11292514, -0.1312699 ,  0.07413186, ..., -0.0345116 ,
          0.09101289,  0.06796468]],

       [[-0.07808004, -0.10946111, -0.09529223, ...,  0.01151106,
          0.09962937, -0.09179376]],

       [[-0.0843792 , -0.10908166, -0.09925203, ...,  0.00971421,
          0.09548534, -0.08486937]],

       [[-0.08389836, -0.10990171, -0.1055292 , ...,  0.00767103,
          0.08909672, -0.07665699]],

       [[-0.08546472, -0.10893059, -0.10001937, ...,  0.00777278,
          0.08489308, -0.07387131]]], dtype=float32)}, dtype=object)


/mnt/storage/gait-0/xin/soccernet_features


## All Inference with 5fps

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_2_0_threads

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
tail -n 2 /mnt/storage/gait-0/xin//logs/$line.log
tail -n 2 /mnt/storage/gait-0/xin//logs/$line/workerlog.0
done

# Rerun Unfinished 5ps inference

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






# Single card inference command

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features


cat inference_matches_todo.txt | while read line 
do 
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

echo "sbatch -p 1080Ti,TitanXx8 --gres=gpu:1 --cpus-per-task 4 -n 1  \
--wrap \"export CUDA_VISIBLE_DEVICES=0; python3.7 main.py --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
--output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
done



         #指定使用的GPU显卡id
python3.7 main.py  --validate -c configs_path/your_config.yaml

1080Ti
watch tail -n 20 /mnt/storage/gait-0/xin//logs/germany_bundesliga.2016-2017.2016-10-22_-_16-30_Ingolstadt_3_-_3_Dortmund.2_LQ.log
watch tail -n 20 /mnt/storage/gait-0/xin//logs/germany_bundesliga.2016-2017.2016-10-22_-_16-30_Ingolstadt_3_-_3_Dortmund.2_LQ/workerlog.0

TitanXx8
watch tail -n 20 /mnt/storage/gait-0/xin//logs/spain_laliga.2016-2017.2016-11-27_-_22-45_Real_Sociedad_1_-_1_Barcelona.1_LQ.log
watch tail -n 20 /mnt/storage/gait-0/xin//logs/spain_laliga.2016-2017.2016-11-27_-_22-45_Real_Sociedad_1_-_1_Barcelona.1_LQ/workerlog.0


watch tail -n 20 /mnt/storage/gait-0/xin//logs/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_LQ/workerlog.0
watch tail -n 20 /mnt/storage/gait-0/xin//logs/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.2_LQ/workerlog.0

python data/soccernet_dense_anchors/check_unfinished_inference.py


/mnt/storage/gait-0/xin//logs/spain_laliga.2015-2016.2016-04-02_-_21-30_Barcelona_1_-_2_Real_Madrid.2_LQ/workerlog.0


/mnt/storage/gait-0/xin//logs/germany_bundesliga.2015-2016.2016-04-16_-_19-30_Bayern_Munich_3_-_0_Schalke.1_LQ/workerlog.0



grep -B 10 -A 10 --color 'Goal' "/mnt/big/multimodal_sports/soccer/SoccerNetv2/spain_laliga/2016-2017/2017-05-21 - 21-00 Malaga 0 - 2 Real Madrid/Labels-v2.json" 


cat labels.txt | while read line
do
echo $line
grep -B 10 -A 10 --color 'Goal' "$line"
done



rsync -a xin@asimov-0-log.svail.baidu.com:"/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/italy_serie-a/2016-2017/2016-11-20\ -\ 22-45\ AC\ Milan\ 2\ -\ 2\ Inter/" ~/Downloads


rsync -a xin@asimov-0-log.svail.baidu.com:"/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2015-2016/2016-04-02\ -\ 21-30\ Barcelona\ 1\ -\ 2\ Real\ Madrid" "/Users/zhouxin16/Downloads/2016-04-02 - 21-30 Barcelona 1 - 2 Real Madrid"


rsync -a xin@asimov-0-log.svail.baidu.com:"/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/england_epl/2016-2017/2017-01-15\ -\ 19-00\ Manchester\ United\ 1\ -\ 1\ Liverpool" "/Users/zhouxin16/Downloads/2017-01-15 - 19-00 Manchester United 1 - 1 Liverpool"





CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_debug
line=spain_laliga.2014-2015.2015-04-25_-_17-00_Espanyol_0_-_2_Barcelona.1_LQ
INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/debug_spain_laliga.2014-2015.2015-04-25_-_17-00_Espanyol_0_-_2_Barcelona.1_LQ  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG


The sampler is not thread safe, cannot use multiple workers
less than 2 hours per file
doesn't each thread have its own?

maybe we need to add data to test one file

try initial model


python data/soccernet_dense_anchors/check_unfinished_inference.py \
--inference_root /mnt/storage/gait-0/xin/soccernet_features_2_0_threads




Average mAP:  0.008546825782252895
Average mAP per class:  [0.0006031103177731429, 0.007023355787837999, 0.004442226162267486, 0.007612015409374928, 0.005620470478412588, 0.0, 0.0, 0.022223125653777934, 0.0, 0.05157126489204886, 0.03267476204242374, 0.0, 0.0, 0.01352570755438252, 0.0, 0.0, 0.0]
Average mAP visible:  0.009282596566468881
Average mAP visible per class:  [0.0031938021994840178, 0.007982699414182792, 0.005258245756984297, 0.012869323746530091, 0.006747150789788531, 0.0, 0.0, 0.022631286871877432, 0.0, 0.052886391047441084, 0.032617970864093654, 0.0, 0.0, 0.013617270939589114, 0.0, 0.0, 0.0]
Average mAP unshown:  0.01031475940999797
Average mAP unshown per class:  [0.0, 0.006847089780842783, 0.0, 0.0024225147560411023, 0.0014923403066344537, 0.0, 0.0, 0.022276967845364672, 0.0, 0.04858405039850779, 0.038938832612243846, 0.0, 0.0, 0.013530076630338976, 0.0, 0.0, 0.0]


rsync -a xin@asimov-0-log.svail.baidu.com:/mnt/storage/gait-0/xin/soccernet_features_debug/spain_laliga.2014-2015.2015-04-25_-_17-00_Espanyol_0_-_2_Barcelona.1_LQ/ /Users/zhouxin16/Downloads/soccernet_features_debug

rsync -a xin@asimov-0-log.svail.baidu.com:/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_5fps/spain_laliga.2014-2015.2015-04-25_-_17-00_Espanyol_0_-_2_Barcelona.1_LQ.mkv /Users/zhouxin16/Downloads/soccernet_features_debug