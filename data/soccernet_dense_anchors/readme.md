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
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_10/ppTimeSformer_dense_event_lr_100_fc_lr_10_epoch_00006.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_3_game_start_offset

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


## try 3 crop to see if I can increase performance

### center crop
    CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_center_crop

    for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
    do 
    line=`basename "$FILE" .mkv`
    INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

    echo "rm /mnt/storage/gait-0/xin//logs/$line.log"
    echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
    --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
    --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
    done > unfinished_inference_center_crop.sh

    python data/soccernet_dense_anchors/check_unfinished_inference.py \
    --inference_root /mnt/storage/gait-0/xin/soccernet_features_6_center_crop  > inference_matches_todo.txt
#### Rerun unfinished

        CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
        INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
        INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_center_crop

        cat inference_matches_todo.txt | while read line
        do
        INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
        INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

        echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
        --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
        --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
        done > unfinished_inference_center_crop.sh

#### Test features

    python data/soccernet_dense_anchors/evaluate_dense_anchors.py \
    --features_root /mnt/storage/gait-0/xin/soccernet_features_6_center_crop \
    --result_jsons_root /mnt/storage/gait-0/xin/soccernet_features_3crops_result_jsons/ \
    --crop center

    python -u src/main.py --SoccerNet_path=/mnt/storage/gait-0/xin/soccernet_features_3crops_result_jsons/ --model_name=pptimesformer_center_crop --features features_array.npy --framerate 1 --LR 0.0001 --head_mode fc --window_size 7 --NMS_window 15 2>&1 | tee -a center_crop.0.0001.log

##### 2 crops


    python -u src/main.py --SoccerNet_path=/mnt/storage/gait-0/xin/soccernet_features_3crops_result_jsons/ --model_name=pptimesformer_2crops --features features_array.npy --features_list features_array.npy,features_array_left.npy --framerate 1 --LR 0.0001 --head_mode fc --window_size 7 --NMS_window 15 2>&1 | tee -a 2crops.0.0001.log

### left crop

    CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_left_crop

    for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
    do 
    line=`basename "$FILE" .mkv`
    INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

    echo "rm /mnt/storage/gait-0/xin//logs/$line.log"
    echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
    --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference_left_crop.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
    --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
    done > unfinished_inference_left_crop.sh

    python data/soccernet_dense_anchors/check_unfinished_inference.py \
    --inference_root /mnt/storage/gait-0/xin/soccernet_features_6_left_crop  > inference_matches_todo.txt

### check all inference dir    
    CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_left_crop

    for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
    do 
    line=`basename "$FILE" .mkv`
    INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

    echo $INFERENCE_DIR
    done > inference_dirs_left_crop.txt

    python data/soccernet_dense_anchors/check_unfinished_inference.py \
    --inference_root /mnt/storage/gait-0/xin/soccernet_features_6_right_crop \
    --inference_dirs_file inference_dirs_left_crop.txt > inference_matches_todo.txt

#### Rerun unfinished

        CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
        INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
        INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_left_crop

        cat inference_matches_todo.txt | while read line
        do
        INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
        INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

        echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
        --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference_left_crop.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
        --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
        done > unfinished_inference_left_crop.sh

#### Test features

        python data/soccernet_dense_anchors/evaluate_dense_anchors.py \
        --features_root /mnt/storage/gait-0/xin/soccernet_features_6_left_crop \
        --result_jsons_root /mnt/storage/gait-0/xin/soccernet_features_3crops_result_jsons/ \
        --crop left
### right crop

    CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_right_crop

    for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
    do 
    line=`basename "$FILE" .mkv`
    INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

    echo "rm /mnt/storage/gait-0/xin//logs/$line.log"
    echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
    --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference_right_crop.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
    --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
    done > unfinished_inference_right_crop.sh

    python data/soccernet_dense_anchors/check_unfinished_inference.py \
    --inference_root /mnt/storage/gait-0/xin/soccernet_features_6_right_crop  > inference_matches_todo.txt

### check all inference dir    
    CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_right_crop

    for FILE in /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/*; 
    do 
    line=`basename "$FILE" .mkv`
    INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

    echo $INFERENCE_DIR
    done > inference_dirs_right_crop.txt

    python data/soccernet_dense_anchors/check_unfinished_inference.py \
    --inference_root /mnt/storage/gait-0/xin/soccernet_features_6_right_crop \
    --inference_dirs_file inference_dirs_right_crop.txt > inference_matches_todo.txt

#### Rerun unfinished

        CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
        INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams
        INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_right_crop

        cat inference_matches_todo.txt | while read line
        do
        INFERENCE_JSON_CONFIG=$CONFIG_DIR/$line.mkv
        INFERENCE_DIR=$INFERENCE_DIR_ROOT/$line

        echo "sbatch -p 1080Ti,1080Ti_slong --exclude asimov-157 --gres=gpu:1 --cpus-per-task 4 -n 1  \
        --wrap \"python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/$line  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference_right_crop.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG\" \
        --output=\"/mnt/storage/gait-0/xin//logs/$line.log\" "
        done > unfinished_inference_right_crop.sh

        bash unfinished_inference_right_crop.sh

#### Test features

        python data/soccernet_dense_anchors/evaluate_dense_anchors.py \
        --features_root /mnt/storage/gait-0/xin/soccernet_features_6_right_crop \
        --result_jsons_root /mnt/storage/gait-0/xin/soccernet_features_3crops_result_jsons/ \
        --crop right


## Check inference logs

CONFIG_DIR=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists_5fps/
INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00028.pdparams
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features
INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features_6_left_crop

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

# Check unfinished
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

looks like '/mnt/data/kangle/datasets/SoccerNetv2_features/england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/1.mkv' is low resolution and is offset by gamestart

/mnt/data/kangle/datasets/SoccerNetv2_features/england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/1_ResNET_TF2.npy


a.shape
(5399, 2048)

original video HQ 3882
1.mkv       DURATION        : 00:45:00.003000000 2700


/mnt/data/kangle/datasets/SoccerNetv2_features/england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/1_TPNs0_1fps.npy 
(2700, 2048)

inference has to be done on low quality


/mnt/storage/gait-0/xin/soccernet_features_3_game_start_offset/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_LQ



features_root = '/mnt/storage/gait-0/xin/soccernet_features_4_ppTimeSformer/'


Average mAP:  0.09007290453289639
Average mAP per class:  [0.2720474159093795, 0.12153369206782262, 0.036611412793572766, 0.24408286481055408, 0.014652519534988516, 0.0, 0.0, 0.2240384262459937, 0.0, 0.2785389094882098, 0.03884592489740687, 0.0, 0.0, 0.30088821131131077, 0.0, 0.0, 0.0]
Average mAP visible:  0.10438751417635218
Average mAP visible per class:  [0.3458375844630926, 0.20291783482882622, 0.03915415905982397, 0.29339214977841765, 0.016429031990266036, 0.0, 0.0, 0.2509649531984421, 0.0, 0.2803347900642102, 0.03894049868445526, 0.0, 0.0, 0.30661673893045305, 0.0, 0.0, 0.0]
Average mAP unshown:  0.07134054404467563
Average mAP unshown per class:  [0.0, 0.10666052510019895, 0.0, 0.07151678962836137, 0.00629467122281144, 0.0, 0.0, 0.2027338799026472, 0.0, 0.23936296047231195, 0.034630049890725816, 0.0, 0.0, 0.2662281963637266, 0.0, 0.0, 0.0]


ffmpeg -i "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/00.worldcup1.mp4" -vf "scale=554x256,fps=5" -map 0:v -c:v libx264 "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/00.worldcup1.5fps.mp4" -y
ffmpeg -i "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/01.Spain-Goal5-closeup.mp4" -vf "scale=564x256,fps=5" -map 0:v -c:v libx264 "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/01.Spain-Goal5-closeup.5fps.mp4" -y
ffmpeg -i "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/02.Spain-Goal5-broadcast_view.mp4" -vf "scale=564x256,fps=5" -map 0:v -c:v libx264 "/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/02.Spain-Goal5-broadcast_view.5fps.mp4" -y

mkdir /mnt/storage/gait-0/xin//logs/00.worldcup1
python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/00.worldcup1.log  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_one_file_inference.yaml -w output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams -o inference_dir=/mnt/storage/gait-0/xin//logs/00.worldcup1 -o DATASET.test.file_path=/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/00.worldcup1.json

/mnt/storage/gait-0/xin//logs/00.worldcup1/features.npy


python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/01.Spain-Goal5-closeup.log  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_one_file_inference.yaml -w output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams -o inference_dir=/mnt/storage/gait-0/xin//logs/01.Spain-Goal5-closeup -o DATASET.test.file_path=/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/01.Spain-Goal5-closeup.json


/mnt/storage/gait-0/xin//logs/02.Spain-Goal5-broadcast_view/features.npy
python3.7 -B -m paddle.distributed.launch --gpus='0' --log_dir=/mnt/storage/gait-0/xin//logs/02.Spain-Goal5-broadcast_view.log  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_one_file_inference.yaml -w output/ppTimeSformer_dense_event_lr_100_fc_lr_100/ppTimeSformer_dense_event_lr_100_fc_lr_100_epoch_00038.pdparams -o inference_dir=/mnt/storage/gait-0/xin//logs/02.Spain-Goal5-broadcast_view -o DATASET.test.file_path=/mnt/storage/gait-0/xin/dev/PaddleVideo/logs/02.Spain-Goal5-broadcast_view.json


2022-11-23 23:31:52,943 [MainThread  ] [INFO ]  Best Performance at end of training 
2022-11-23 23:31:52,943 [MainThread  ] [INFO ]  a_mAP visibility all: 0.576915737029858
2022-11-23 23:31:52,943 [MainThread  ] [INFO ]  a_mAP visibility all per class: [0.7338447762386684, 0.5816676236687317, 0.7785980815311692, 0.6822690267005584, 0.3760402573240339, 0.4928297533487119, 0.5222437585527825, 0.6015392146933541, 0.8139629113582744, 0.772587339787498, 0.7518053192678322, 0.4991151557316356, 0.6624030206023768, 0.8673679774094701, 0.603418822670661, 0.02756402346331268, 0.04031046715851462]
2022-11-23 23:31:52,943 [MainThread  ] [INFO ]  a_mAP visibility visible: 0.6361439291972443
2022-11-23 23:31:52,944 [MainThread  ] [INFO ]  a_mAP visibility visible per class: [0.760193110495957, 0.7747634876492324, 0.7888214837989929, 0.7617382796254766, 0.4016478554140963, 0.49547309120115374, 0.5239630298264336, 0.732484378839268, 0.816210900339387, 0.8079184815799014, 0.756169468497864, 0.5120977396697852, 0.7214378152654716, 0.8719719240855749, 0.6394084783316006, 0.24309512085154883, 0.207052150881406]
2022-11-23 23:31:52,944 [MainThread  ] [INFO ]  a_mAP visibility unshown: 0.3638725516259772
2022-11-23 23:31:52,944 [MainThread  ] [INFO ]  a_mAP visibility unshown per class: [0.0, 0.47558556016408116, 0.0, 0.26110491964466687, 0.19200377423112605, 0.030872993255438394, 0.01619923815464124, 0.45325791842697494, 0.7517355800425992, 0.6646306699993666, 0.3864525770855391, 0.4798547913349795, 0.048492404412619275, 0.8306191083040396, 0.13953363608163166, 0.0, 0.0]
2022-11-23 23:31:52,946 [MainThread  ] [INFO ]  Checking/Download features and labels locally



sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1 --wrap "python -u -B -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_balanced main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_balanced.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" --output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_balanced.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW_balanced main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW_balanced.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.00001_sgd_60_random_scale_adamW_balanced.log"



sbatch -p V100_GAIT --nodelist=asimov-227 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW_balanced main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW_balanced.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.0001_sgd_60_random_scale_adamW_balanced.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW_balanced main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW_balanced.yaml -w pretrained_weights/ppTimeSformer_k400_16f_distill.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60_random_scale_adamW_balanced.log"

python data/soccernet_dense_anchors/evaluate_dense_anchors.py

balanced actually worse:

2022-12-02 02:46:49,910 [MainThread  ] [INFO ]  Best Performance at end of training 
2022-12-02 02:46:49,911 [MainThread  ] [INFO ]  a_mAP visibility all: 0.45982091034791645
2022-12-02 02:46:49,911 [MainThread  ] [INFO ]  a_mAP visibility all per class: [0.5189455663377526, 0.4116411575609777, 0.5880927903728196, 0.609399005210963, 0.23152746232902469, 0.36884666421184314, 0.4122902373058888, 0.5064612133815835, 0.7122760259870689, 0.6418326668276235, 0.6302575242796522, 0.38493336861830596, 0.5283362989581609, 0.7939564760897224, 0.39278392028149234, 0.06413417531586627, 0.0212409228458336]
2022-12-02 02:46:49,912 [MainThread  ] [INFO ]  a_mAP visibility visible: 0.5281647370617496
2022-12-02 02:46:49,912 [MainThread  ] [INFO ]  a_mAP visibility visible per class: [0.6341910900817699, 0.5398507082992499, 0.6045171382405693, 0.6902625720741247, 0.2553067332372326, 0.36991745516849556, 0.4156381988614815, 0.6616281828231739, 0.7142213324984563, 0.6728101192987885, 0.6348331943757354, 0.36737920082029474, 0.605935964119121, 0.8045367456042793, 0.42615260019488616, 0.4608747667195578, 0.12074452763252676]
2022-12-02 02:46:49,912 [MainThread  ] [INFO ]  a_mAP visibility unshown: 0.2909832648010812
2022-12-02 02:46:49,912 [MainThread  ] [INFO ]  a_mAP visibility unshown per class: [0.0, 0.3599121128738172, 0.0, 0.20664835959393435, 0.07352151339243154, 0.03678496191865176, 0.023163776044697947, 0.3417160855078865, 0.664361284648191, 0.5586421840208868, 0.31106841531289964, 0.406840057575185, 0.02064045190171563, 0.7009065438446038, 0.0785766957791549, 0.0, 0.0]
2022-12-02 02:46:49,914 [MainThread  ] [INFO ]  Checking/Download features and labels locally
  0%|                                                                                              



sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_HowTo100M_pretrained.log"

lr 0.0025 did not work


sbatch -p V100_GAIT --nodelist=asimov-227 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01 main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.0001.yaml main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.0001.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.0001.log"


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.005.yaml main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.005.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.005.log"



sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_howto100M_lr_0.0025.yaml main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_howto100M_lr_0.0025.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_howto100M_lr_0.0025"




sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_k600_lr_0.0025.yaml main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_k600_lr_0.0025.yaml -w pretrained_weights/pretrained_weights/TimeSformer_divST_8x32_224_K600.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_10_k600_lr_0.0025"


data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01_sgd.yaml


sbatch -p V100_GAIT --nodelist=asimov-228 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01_sgd.yaml main.py --validate -c data/soccernet/experiments/pptimesformer/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01_sgd.yaml -w pretrained_weights/TimeSformer_divST_32x32_224_HowTo100M.pdparams" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_pptimesformer_k400_videos_dense_event_lr_100_fc_lr_multiplier_100_howto100M_lr_0.01_sgd.log"


sgd was not the issue

