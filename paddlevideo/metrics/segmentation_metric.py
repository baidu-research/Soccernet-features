# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")

def argrelmax(prob, threshold=0.7):
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(
            p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


@METRIC.register
class SegmentationMetric(BaseMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_interval=1,
                 tolerance=5,
                 boundary_threshold=0.7):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        # cls score
        self.overlap = overlap
        self.overlap_len = len(overlap)

        self.total_tp = np.zeros(self.overlap_len)
        self.total_fp = np.zeros(self.overlap_len)
        self.total_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

        # boundary score
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance
        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        groundTruth = data[1]

        outputs_np = outputs.cpu().detach().numpy()
        gt_np = groundTruth.cpu().detach().numpy()[0, :]

        recognition = []
        for i in range(outputs_np.shape[0]):
            recognition = np.concatenate((recognition, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(outputs_np[i])]
            ]))
        recog_content = list(recognition)

        gt_content = []
        for i in range(gt_np.shape[0]):
            gt_content = np.concatenate((gt_content, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(gt_np[i])]
            ]))
        gt_content = list(gt_content)

        # cls metric
        tp, fp, fn = np.zeros(self.overlap_len), np.zeros(
            self.overlap_len), np.zeros(self.overlap_len)

        correct = 0
        total = 0
        edit = 0

        for i in range(len(gt_content)):
            total += 1
            #accumulate
            self.total_frame += 1

            if gt_content[i] == recog_content[i]:
                correct += 1
                #accumulate
                self.total_correct += 1

        edit_num = edit_score(recog_content, gt_content)
        edit += edit_num
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, self.overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

            # accumulate
            self.total_tp[s] += tp1
            self.total_fp[s] += fp1
            self.total_fn[s] += fn1

        # accumulate
        self.total_video += 1

        # boundary metric
        # ignore invalid frames

        pred_idx = argrelmax(outputs_np, threshold=self.boundary_threshold)
        gt_idx = argrelmax(gt_np, threshold=self.boundary_threshold)

        n_frames = outputs_np.shape[0]
        tp = 0.0
        fp = 0.0
        fn = 0.0

        hits = np.zeros(len(gt_idx))

        # calculate true positive, false negative, false postive, true negative
        for i in range(len(pred_idx)):
            dist = np.abs(np.array(gt_idx) - pred_idx[i])
            min_dist = np.min(dist)
            idx = np.argmin(dist)

            if min_dist <= self.tolerance and hits[idx] == 0:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1

        fn = len(gt_idx) - sum(hits)
        tn = n_frames - tp - fp - fn

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.n_frames += n_frames
        self.n_correct += tp + tn

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        # cls metric
        Acc = 100 * float(self.total_correct) / self.total_frame
        Edit = (1.0 * self.total_edit) / self.total_video
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.total_tp[s] / float(self.total_tp[s] +
                                                 self.total_fp[s])
            recall = self.total_tp[s] / float(self.total_tp[s] +
                                              self.total_fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # boundary meric
        # accuracy
        boundary_Acc = 100 * self.n_correct / self.n_frames

        # Boudnary F1 Score
        b_precision = self.tp / float(self.tp + self.fp)
        b_recall = self.tp / float(self.tp + self.fn)

        f1s = 2.0 * (b_precision * b_recall) / (b_precision + b_recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100

        # log metric
        log_mertic_info = "dataset model performence: "
        # preds ensemble
        log_mertic_info += "Cls_Acc: {:.4f}, ".format(Acc)
        log_mertic_info += 'Edit: {:.4f}, '.format(Edit)
        for s in range(len(self.overlap)):
            log_mertic_info += 'F1@{:0.2f}: {:.4f}, '.format(self.overlap[s],
                                                    Fscore[self.overlap[s]])
        log_mertic_info += "Boundary_Acc: {:.4f}, ".format(boundary_Acc)
        log_mertic_info += "Boundary_precision: {:.4f}, ".format(b_precision * 100)
        log_mertic_info += "Boundary_recall: {:.4f}, ".format(b_recall * 100)
        log_mertic_info += "Boundary_F1score: {:.4f}.".format(f1s)
        logger.info(log_mertic_info)

        # log metric
        metric_dict = dict()
        metric_dict['Cls_Acc'] = Acc
        metric_dict['Edit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['F1@{:0.2f}'.format(self.overlap[s])] = Fscore[self.overlap[s]]
        metric_dict['Boundary_Acc'] = boundary_Acc
        metric_dict['Boundary_precision'] = b_precision * 100
        metric_dict['Boundary_recall'] = b_recall * 100
        metric_dict['Boundary_F1score'] = f1s

        # clear for next epoch
        # cls
        self.total_tp = np.zeros(self.overlap_len)
        self.total_fp = np.zeros(self.overlap_len)
        self.total_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0
        # boundary
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0
        
        return metric_dict
