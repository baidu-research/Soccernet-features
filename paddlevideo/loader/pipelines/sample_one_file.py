# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
from PIL import Image
try:
    import SimpleITK as sitk
except ImportError as e:
    print(
        f"Warning! {e}, [SimpleITK] package and it's dependencies is required for PP-Care."
    )
import cv2

from ..registry import PIPELINES

def clip_video_start_end(start_secs, end_secs, length_secs):
    sample_length_secs = end_secs - start_secs
    if start_secs < 0:
        return (0, sample_length_secs)
    if end_secs >= length_secs:
        return (length_secs - 1 - sample_length_secs, length_secs - 1)
    return (start_secs, end_secs)

@PIPELINES.register()
class OneFileSampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 num_seg,
                 seg_len,
                 frame_interval=None,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                #  linspace_sample=False,
                 use_pil=True,
                 sample_length_secs = 1.0):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.frame_interval = frame_interval
        self.valid_mode = valid_mode
        self.select_left = select_left
        self.dense_sample = dense_sample
        # self.linspace_sample = linspace_sample
        self.use_pil = use_pil
        self.sample_length_secs = sample_length_secs

    def _get(self, frames_idx, results):
        data_format = results['format']

        if data_format == "frame":
            frame_dir = results['frame_dir']
            imgs = []
            for idx in frames_idx:
                img = Image.open(
                    os.path.join(frame_dir,
                                 results['suffix'].format(idx))).convert('RGB')
                imgs.append(img)

        elif data_format == "MRI":
            frame_dir = results['frame_dir']
            imgs = []
            MRI = sitk.GetArrayFromImage(sitk.ReadImage(frame_dir))
            for idx in frames_idx:
                item = MRI[idx]
                item = cv2.resize(item, (224, 224))
                imgs.append(item)

        elif data_format == "video":
            if results['backend'] == 'cv2':
                frames = np.array(results['frames'])
                imgs = []
                for idx in frames_idx:
                    imgbuf = frames[idx]
                    img = Image.fromarray(imgbuf, mode='RGB')
                    imgs.append(img)
            elif results['backend'] == 'decord':
                container = results['frames']
                if self.use_pil:
                    frames_select = container.get_batch(frames_idx)
                    # dearray_to_img
                    np_frames = frames_select.asnumpy()
                    imgs = []
                    for i in range(np_frames.shape[0]):
                        imgbuf = np_frames[i]
                        imgs.append(Image.fromarray(imgbuf, mode='RGB'))
                else:
                    # print('frames_idx', frames_idx)
                    # if frames_idx.ndim != 1:
                    #     frames_idx = np.squeeze(frames_idx)
                    frames_idx = np.array(frames_idx)
                    frame_dict = {
                        idx: container[idx].asnumpy()
                        for idx in np.unique(frames_idx)
                    }
                    imgs = [container[idx].asnumpy() for idx in frames_idx]
            elif results['backend'] == 'pyav':
                imgs = []
                frames = np.array(results['frames'])
                for idx in frames_idx:
                    if self.dense_sample:
                        idx = idx - 1
                    imgbuf = frames[idx]
                    imgs.append(imgbuf)
                imgs = np.stack(imgs)  # thwc
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        results['imgs'] = imgs

        # print('results.keys()', results.keys())
        # for key in ['filename', 'labels', 'frames', 'frames_len']:
        #     print('results', key, results[key])
        # print('len(imgs)', len(imgs))

        return results

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        frames_idx = []

        start_secs = results['start_secs']
        end_secs = results['end_secs']
        fps = results['fps']

        start_idx = int(start_secs * fps) # only off by 1 frame, it's OK
        end_idx = int(end_secs * fps)
        if end_idx == frames_len:
            end_idx = frames_len - 1

        results['start_idx'] = start_idx
        results['end_idx'] = end_idx

        results['start_secs'] = start_secs
        results['end_secs'] = end_secs

        frames_idx = np.linspace(start_idx, end_idx,
                                    self.num_seg).astype(np.int64)
        return self._get(frames_idx, results)