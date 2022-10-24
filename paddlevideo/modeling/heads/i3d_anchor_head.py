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
import paddle.nn as nn
from paddle import ParamAttr

from ..registry import HEADS
from ..weight_init import weight_init_
from .base import BaseHead
import paddle.nn.functional as F

@HEADS.register()
class I3DAnchorHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(name='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        drop_ratio (float): Probability of dropout layer. Default: 0.5.
        std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name='DenseAnchorLoss'),
                 spatial_type='avg',
                 drop_ratio=0.5,
                 std=0.01,
                 event_time_loss_weight = 1.0,
                 output_mode = None,
                 events_lr_multiplier = 1.0,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cfg, **kwargs)

        self.spatial_type = spatial_type
        self.drop_ratio = drop_ratio
        self.stdv = std
        self.event_time_loss_weight = event_time_loss_weight
        self.output_mode = output_mode
        if self.drop_ratio != 0:
            self.dropout = nn.Dropout(p=self.drop_ratio)
        else:
            self.dropout = None
        self.fc = nn.Linear(
            self.in_channels,
            self.num_classes,
            weight_attr=ParamAttr(learning_rate=10.0),
            bias_attr=ParamAttr(learning_rate=10.0),
        )

        self.fc_event_times = nn.Linear(
            self.in_channels,
            self.num_classes,
            weight_attr=ParamAttr(learning_rate=events_lr_multiplier),
            bias_attr=ParamAttr(learning_rate=events_lr_multiplier),
        )

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3D((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        weight_init_(self.fc, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)
        weight_init_(self.fc_event_times, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        N = paddle.shape(x)[0]
        x = x.reshape([N, -1])
        # [N, in_channels]
        cls_score = self.fc(x)
        event_times = F.sigmoid(self.fc_event_times(x)) # need to normalize this to be between 0 and 1
        # [N, num_classes]
        if self.output_mode == 'features':
            return x
        # probability_time mode
        else:
            return cls_score, event_times

    # Override loss for our dual loss case here
    def loss(self, cls_score, class_labels, event_times, event_time_labels, valid_mode=False, if_top5=True, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

        """

        losses = self.loss_func(cls_score, class_labels, event_times, event_time_labels, event_time_loss_weight = self.event_time_loss_weight)
        # print(loss)
        # ddd

        if if_top5:
            top1, top5 = self.get_acc(cls_score, class_labels, valid_mode)
            losses['top1'] = top1
            losses['top5'] = top5
            # losses['loss'] = loss
        else:
            top1 = self.get_acc(cls_score, class_labels, valid_mode, if_top5)
            losses['top1'] = top1
            # losses['loss'] = loss
        return losses