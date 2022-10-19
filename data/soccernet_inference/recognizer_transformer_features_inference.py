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

import paddle
import paddle.nn.functional as F
from paddlevideo.utils import get_logger

from ...registry import RECOGNIZERS
from .base import BaseRecognizer

logger = get_logger("paddlevideo")

@RECOGNIZERS.register()
class RecognizerTransformerFeaturesInference(BaseRecognizer):
    """Transformer's recognizer model framework."""
    def forward_net(self, imgs):
        # imgs.shape=[N,C,T,H,W], for transformer case
        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = imgs
        
        return feature.squeeze()

    def test_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        labels = data_batch[1:]
        feature = self.forward_net(imgs)
        return feature 
