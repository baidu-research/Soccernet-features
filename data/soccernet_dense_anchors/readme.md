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

# List of changed files and corresponding changes.

- Label files processing are changed and labels of category and event_times are composed into dicts to send into the pipeline. Class names are added into the init.
    
        paddlevideo/loader/dataset/video_dense_anchors.py

        paddlevideo/loader/dataset/__init__.py

    Added temporal coordinate embedding to inputs. Removed event time loss for background class.

        paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py

- Added EventSampler

        paddlevideo/loader/pipelines/sample.py

        paddlevideo/loader/pipelines/__init__.py

    Added sampling one whole video file.

        paddlevideo/loader/pipelines/sample_one_file.py

- Multitask losses.

        paddlevideo/modeling/losses/dense_anchor_loss.py
        
        paddlevideo/modeling/losses/__init__.py

- Changed head output. Class and event times.

        paddlevideo/modeling/heads/i3d_anchor_head.py

        paddlevideo/modeling/heads/pptimesformer_anchor_head.py

        paddlevideo/modeling/heads/__init__.py

5. Input and output format in train_step, val step etc.

    paddlevideo/modeling/framework/recognizers/recognizer_transformer_dense_anchors.py
    
    paddlevideo/modeling/framework/recognizers/__init__.py

6. Add a new mode to log class loss and event time loss.

    paddlevideo/utils/record.py

7. Added parser for one video file list.

    paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py

    paddlevideo/loader/dataset/__init__.py

8. Added MODEL.head.name and MODEL.head.output_mode branch to process outputs of class scores and event_times. Also unified feature inference with simple classification mode.

    paddlevideo/tasks/test.py

9. Lower generate lower resolution script.

    data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py

10. Balanced samples do not seem necessary 
    
    data/soccernet_dense_anchors/balance_class_samples.py

11. Collate file to replace the current library file

    /mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/lib/python3.7/site-packages/paddle/fluid/dataloader/collate.py

12. Config files

    data/soccernet/soccernet_videoswin_k400_dense_one_file_inference.yaml

13. Updated to support dense anchors

    data/soccernet/split_annotation_into_train_val_test.py

# Comments

1. TODO paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py can inherit from paddlevideo/loader/dataset/video_dense_anchors.py







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