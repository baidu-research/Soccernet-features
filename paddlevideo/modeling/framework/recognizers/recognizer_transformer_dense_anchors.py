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

from re import A
import paddle
import paddle.nn.functional as F
from paddlevideo.modeling.heads.pptimesformer_head import ppTimeSformerHead
from paddlevideo.utils import get_logger

from ...registry import RECOGNIZERS
from .base import BaseRecognizer

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class RecognizerTransformerDenseAnchors(BaseRecognizer):
    """Transformer's recognizer model framework."""
    def forward_net(self, imgs):
        # imgs.shape=[N,C,T,H,W], for transformer case
        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = imgs

        if self.head is not None:
            cls_score = self.head(feature)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        imgs = data_batch['imgs']
        class_labels = data_batch['label']
        event_time_labels = data_batch['event_time_labels']
        cls_score, event_times = self.forward_net(imgs)

        loss_metrics = self.head.loss(cls_score, class_labels, event_times, event_time_labels)

        # add other loss metrics, like top1, top 5 ...
        return loss_metrics

    def val_step(self, data_batch):
        imgs = data_batch['imgs']
        class_labels = data_batch['label']
        event_time_labels = data_batch['event_time_labels']
        cls_score, event_times = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, class_labels, event_times, event_time_labels, valid_mode=True)
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        # imgs = data_batch[0]
        # num_views = imgs.shape[2] // self.runtime_cfg.test.num_seg
        # cls_score = []
        # for i in range(num_views):
        #     view = imgs[:, :, i * self.runtime_cfg.test.num_seg:(i + 1) *
        #                 self.runtime_cfg.test.num_seg]
        #     cls_score.append(self.forward_net(view))
        # cls_score = self._average_view(cls_score,
        #                                self.runtime_cfg.test.avg_type)
        
        imgs = data_batch['imgs']
        result = self.forward_net(imgs)
        return result

    def infer_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        num_views = imgs.shape[2] // self.runtime_cfg.test.num_seg
        cls_score = []
        for i in range(num_views):
            view = imgs[:, :, i * self.runtime_cfg.test.num_seg:(i + 1) *
                        self.runtime_cfg.test.num_seg]
            cls_score.append(self.forward_net(view))
        cls_score = self._average_view(cls_score,
                                       self.runtime_cfg.test.avg_type)
        return cls_score

    def _average_view(self, cls_score, avg_type='score'):
        """Combine the predicted results of different views

        Args:
            cls_score (list): results of multiple views
            avg_type (str, optional): Average calculation method. Defaults to 'score'.
        """
        assert avg_type in ['score', 'prob'], \
            f"Currently only the average of 'score' or 'prob' is supported, but got {avg_type}"
        if avg_type == 'score':
            return paddle.add_n(cls_score) / len(cls_score)
        elif avg_type == 'prob':
            return paddle.add_n(
                [F.softmax(score, axis=-1)
                 for score in cls_score]) / len(cls_score)
        else:
            raise NotImplementedError
