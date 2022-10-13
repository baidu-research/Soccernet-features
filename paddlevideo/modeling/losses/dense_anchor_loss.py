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

import paddle
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class DenseAnchorLoss(BaseWeightedLoss):
    # def _forward(self, score, labels, **kwargs):
    def forward(self, cls_scores, cls_labels, event_times, event_time_labels, event_time_loss_weight = 1.0):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        # two types of losses, cross_entropy, then l2

        # cls_scores = predictions['class_scores']
        # cls_labels = labels['class_labels']
        # event_times = predictions['event_times']
        # event_time_labels = labels['event_time_labels']

        loss_class = F.cross_entropy(cls_scores, cls_labels)
        # this has to be multiplied by class indices
        # how to pass in a weight coefficient
        # print('event_times', event_times)
        # print('event_time_labels', event_time_labels)
        cls_one_hot = F.one_hot(cls_labels, event_times.shape[1])
        event_times_for_labeled_class = paddle.squeeze(paddle.sum(event_times * cls_one_hot, axis = 2), 1)
        print('event_times_for_labeled_class', event_times_for_labeled_class)
        print('event_time_labels', event_time_labels)

        loss_event_times = F.mse_loss(event_times_for_labeled_class, event_time_labels) # try F.l1_loss?
        # missing a weight argument here, or even make two separate losses?
        loss_total = loss_class + event_time_loss_weight * loss_event_times
        loss = {
            'loss_class': loss_class,
            'loss_event_times': loss_event_times,
            'loss': loss_total}
        # return the component losses?
        return loss
