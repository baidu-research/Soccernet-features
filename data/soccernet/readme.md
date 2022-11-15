# 修改
# 并行，少用glob

# Lower video resolution 

(high resolution is very slow for training)

all change resolution commands
split the commands

This takes a long time, so the script prints out the commands and you can split the commands in order to parallelize.

python data/soccernet/generate_training_short_clips.py --clips_folder=/mnt/storage/gait-0/xin/dataset/soccernet_456x256 > data/soccernet/generate_training_short_clips.sh

<!-- sed -n '0~5p' oldfile > newfile -->

mkdir data/soccernet/short_clips_parallel

for i in {0..399};
do
sed -n ${i}~400p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel/${i}.sh;
done

for i in {0..400};
do
sed -n ${i}~401p data/soccernet/generate_training_short_clips.sh > data/soccernet/short_clips_parallel_401/${i}.sh;
done

## this may need a couple of runs just in case the previous run fails
for i in {0..399};
do
sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
"echo no | bash data/soccernet/short_clips_parallel/${i}.sh" \
--output="data/soccernet/short_clips_parallel/${i}.log"
done

for i in {0..400};
do
sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
"echo no | bash data/soccernet/short_clips_parallel_401/${i}.sh" \
--output="data/soccernet/short_clips_parallel_401/${i}.log"
done

# Generate label files. Write json_list.txt and json files containing labels for each video file
python data/soccernet/generate_labels.py \
--extension mkv \
--raw_videos_root /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
--labels_root /mnt/big/multimodal_sports/soccer/SoccerNetv2 \
--clips_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256

# labels to paddle list

python data/soccernet/labels_to_pdvideo_format.py \
--clips_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256 \
--output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256

# split annotations into train val test
python data/soccernet/split_annotation_into_train_val_test.py \
--annotation_file /mnt/storage/gait-0/xin/dataset/soccernet_456x256/annotation.txt \
--clips_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256 \
--mode text

cp /mnt/storage/gait-0/xin/dataset/soccernet_456x256/train.list .
cp /mnt/storage/gait-0/xin/dataset/soccernet_456x256/val.list .
cp /mnt/storage/gait-0/xin/dataset/soccernet_456x256/test.list .

# pretrained: "pretrained_weights/SwinTransformer_imagenet.pdparams" 

python3.7 -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_videoswin main.py --validate -c data/soccernet/soccernet_videoswin_k400.yaml

pip install -r requirements.txt

python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_videoswin main.py --validate -c data/soccernet/soccernet_videoswin_k400.yaml

# for certain ones with Decoder (codec dvb_teletext) error
ffmpeg -ss 0:33:10 -i "/mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/england_epl/2016-2017/2017-01-15 - 19-00 Manchester United 1 - 1 Liverpool/1_HQ.mkv"                 -vf scale=456x256 -map 0:v -map 0:a -c copy  -c:v libx264 -c:a aac -strict experimental -b:a 98k                 -t 0:00:10 "/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ.0-33-10.2000.10.mkv"


/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ.0-31-40.1910.10.mkv


for i in {0..19};
do
sed -n ${i}~20p data/soccernet/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ.sh > data/soccernet/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ_parallel/${i}.sh;
done

for i in {0..19};
do
sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
"bash data/soccernet/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ_parallel/${i}.sh" \
--output="data/soccernet/england_epl.2016-2017.2017-01-15_-_19-00_Manchester_United_1_-_1_Liverpool.1_HQ_parallel/${i}.log"
done


sbatch -p 1080Ti_slong --exclude asimov-157 --gres=gpu:8 --cpus-per-task 20 -n 1 --wrap \
"bash data/soccernet/run.sh" --output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_1.log"


sbatch -p 1080Ti_slong --exclude asimov-157 --gres=gpu:10 --cpus-per-task 20 -n 1 \
--wrap "bash data/soccernet/run.sh" --output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_3.log"

# changed parameters
batch_size
global_batch_size

# Sampling





num_seg 8 seg_len 1, equally sample 8 frames from the clip

put  "RecognizerTransformerFeaturesInference" #Mandatory, indicate the type of network, associate to the 'paddlevideo/modeling/framework/' 

copy test.py to paddlevideo/tasks/test.py


why -w instead of using config


sbatch -p V100_GAIT --nodelist=asimov-230 --account=gait -t 30-00:00:00 --gres=gpu:8 --cpus-per-task 40 -n 1  \
--wrap "bash data/soccernet/run.sh" \
--output="/mnt/storage/gait-0/xin//logs/soccernet_videoswin_20_num_segs_16.log"


python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_soccernet_feature_test  main.py  --test -c data/soccernet_inference/soccernet_videoswin_k400_extract_features.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams -o features_dir=/mnt/storage/gait-0/xin/dev/PaddleVideo/temp2



python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_soccernet_feature_test  main.py  --test -c data/soccernet_inference/soccernet_videoswin_k400_extract_features.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams -o features_dir=/mnt/storage/gait-0/xin/dev/PaddleVideo/temp2 -o DATASET.test.file_path=inference2.list


END epoch:1   val 
END epoch:1   train

mkdir log_videoswin_11_videoswin/

python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_videoswin_15_videoswin_batch_40 main.py --validate -c data/soccernet/soccernet_videoswin_k400.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams 2>&1 | tee log_videoswin_15_videoswin_batch_40/run.log



/mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/lib/python3.7/site-packages/paddle/fluid/dataloader/collate.py

