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

from paddle.nn import Linear

from ..registry import HEADS
from ..weight_init import trunc_normal_, weight_init_
from .base import BaseHead
from paddle import ParamAttr
from paddle.regularizer import L2Decay
import paddle.nn.functional as F


@HEADS.register()
class ppTimeSformerAnchorHead(BaseHead):
    """TimeSformerHead Head.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        std(float): Std(Scale) value in normal initilizar. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to initialize.

    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name='DenseAnchorLoss'),
                 std=0.02,
                 output_mode = None,
                 event_time_loss_weight = 1.0,
                 events_lr_multiplier = 1.0,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cfg, **kwargs)
        self.std = std
        self.output_mode = output_mode
        self.event_time_loss_weight = event_time_loss_weight
        self.fc = Linear(self.in_channels,
                         self.num_classes,
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.fc_event_times = Linear(
            self.in_channels,
            self.num_classes,
            weight_attr=ParamAttr(learning_rate=events_lr_multiplier),
            bias_attr=ParamAttr(learning_rate=events_lr_multiplier),
        )

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        # NOTE: Temporarily use trunc_normal_ instead of TruncatedNormal
        trunc_normal_(self.fc.weight, std=self.std)

        weight_init_(self.fc_event_times,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        # NOTE: Temporarily use trunc_normal_ instead of TruncatedNormal
        trunc_normal_(self.fc_event_times.weight, std=self.std)

    def forward(self, x):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # XXX: check dropout location!
        # x.shape = [N, embed_dim]

        if self.output_mode == 'features':
            return x
        # probability_time mode
        else:
            cls_score = self.fc(x)
            event_times = F.sigmoid(self.fc_event_times(x)) # need to normalize this to be between 0 and 1

            return cls_score, event_times

        # score = self.fc(x)
        # # [N, num_class]
        # # x = F.softmax(x)  # NOTE remove
        # return score

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