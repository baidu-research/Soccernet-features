# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp
import copy
import random
import numpy as np
import json

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger
import paddle

logger = get_logger("paddlevideo")


def add_coordinates_embedding_to_imgs(results):
    b = results['imgs'].shape[0]
    t = results['imgs'].shape[1]
    h = results['imgs'].shape[2]
    w = results['imgs'].shape[3]

    # add coordinate embedding for better regression
    coordinate_array = np.linspace(-0.5, 0.5, t)
    coordinate_array = coordinate_array.reshape(1,t,1,1)
    # coordinate_array = np.tile(coordinate_array, (3,1,2,2))
    # print(coordinate_array)
    coordinate_array = np.tile(coordinate_array, (b,1,h,w))
    if isinstance(results['imgs'], paddle.Tensor):
        # results['imgs'] = results['imgs'] + paddle.to_tensor(coordinate_array)
        # no coordinate array in pptimesformermode yet
        pass
    else: #nd.array
        results['imgs'] += coordinate_array
    return

@DATASETS.register()
class VideoDenseAnchorsDataset(BaseDataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
       .. code-block:: txt
           path/000.mp4 1
           path/001.mp4 1
           path/002.mp4 2
           path/003.mp4 2
       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.
    """
    def __init__(self, file_path, pipeline, num_retries=5, suffix='', **kwargs):
        self.num_retries = num_retries
        self.suffix = suffix
        super().__init__(file_path, pipeline, **kwargs)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                if len(line.strip()) == 0:
                    continue
                line_json = json.loads(line)

                #TODO(hj): Required suffix format: may mp4/avi/wmv
                # filename = filename + self.suffix
                if self.data_prefix is not None:
                    filename = line_json['filename']
                    line_json['filename'] = osp.join(self.data_prefix, filename)
                info.append(line_json)
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)

                # Sample results:
                # {'filename': '/mnt/storage/gait-0/xin/dataset/soccernet_456x256/england_epl.2014-2015.2015-02-21_-_18-00_Chelsea_1_-_1_Burnley.1_HQ.0-00-00.0.10.mkv', 
                # 'label': 0, 'event_time': 2.0, 'clip_length_secs': 10.0, 'format': 'video', 'backend': 'decord', 
                # 'frames': <decord.video_reader.VideoReader object at 0x7f4d2affd0b8>, 'frames_len': 250, 
                # 'random_offset': 1.283822639752263, 'start_secs_0': 0.783822639752263, 'end_secs_0': 5.7838226397522625, 'start_secs_clipped': 0.783822639752263, 'end_secs_clipped': 5.7838226397522625, 
                # 'start_idx': 20, 'end_idx': 145, 'start_secs': 0.783822639752263, 'end_secs': 5.7838226397522625, 'event_time_in_sampled_clip_fraction': 0.2432354720495474, 
                # 'imgs': array([[[[-0.6109256 , -0.55955136, -0.6109256 , ..., -0.57667613,

            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            
            # print(results['imgs'][0,:,3,3])
            add_coordinates_embedding_to_imgs(results)
            # print(results['imgs'][0,:,3,3])
            # ddd
            # print(results['imgs'].shape)
            # (3, 16, 224, 224)

            # b = results['imgs'].shape[0]
            # t = results['imgs'].shape[1]
            # h = results['imgs'].shape[2]
            # w = results['imgs'].shape[3]

            # # add coordinate embedding for better regression
            # coordinate_array = np.linspace(-0.5, 0.5, t)
            # coordinate_array = coordinate_array.reshape(1,t,1,1)
            # # coordinate_array = np.tile(coordinate_array, (3,1,2,2))
            # # print(coordinate_array)
            # coordinate_array = np.tile(coordinate_array, (b,1,h,w))
            # results['imgs'] += coordinate_array

            return {
                'imgs': results['imgs'], 
                'label': np.array([results['label']]),
                'event_time_labels': np.array(results['event_time_in_sampled_clip_fraction'], dtype = np.float32)}

    def prepare_test(self, idx):
        """TEST. Prepare the data for test given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            
            add_coordinates_embedding_to_imgs(results)

            return {
                'imgs': results['imgs'], 
                'label': np.array([results['label']]),
                'event_time_labels': np.array(results['event_time_in_sampled_clip_fraction'], dtype = np.float32)}
