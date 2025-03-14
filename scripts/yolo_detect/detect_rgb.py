import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from .nets import nn
from .utils import util
from .utils.dataset import Dataset
import cv2

input_size = 640

def pre_load_model(weidht_path):

    model = torch.load(weidht_path, map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

def yolo_detect(frame,weidht_path):
    model = torch.load(weidht_path, map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

    image = frame.copy()
    shape = image.shape[:2]  # current shape [height, width]
    r = min(1.0, input_size / shape[0], input_size / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = input_size - pad[0]
    h = input_size - pad[1]
    w = numpy.mod(w, stride)
    h = numpy.mod(h, stride)
    w /= 2
    h /= 2
    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image,
                               top, bottom,
                               left, right,
                               cv2.BORDER_CONSTANT)  # add border
    # Convert HWC to CHW, BGR to RGB
    image = image.transpose((2, 0, 1))[::-1]
    image = numpy.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze(dim=0)
    image = image.cuda()
    image = image.half()
    image = image / 255
    # Inference
    outputs = model(image)
    # NMS
    outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)

    for output in outputs:
        output = output.clone()
        if len(output):
            box_output = output[:, :6]
            kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
        else:
            box_output = output[:, :6]
            kps_output = output[:, 6:]

        r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

        box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
        box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
        box_output[:, :4] /= r

        box_output[:, 0].clamp_(0, shape[1])  # x
        box_output[:, 1].clamp_(0, shape[0])  # y
        box_output[:, 2].clamp_(0, shape[1])  # x
        box_output[:, 3].clamp_(0, shape[0])  # y

        kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
        kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
        kps_output[..., 0] /= r
        kps_output[..., 1] /= r
        kps_output[..., 0].clamp_(0, shape[1])  # x
        kps_output[..., 1].clamp_(0, shape[0])  # y

        # for box in box_output:
        #     box = box.cpu().numpy()

        kps = []
        for kpt in reversed(kps_output):
            for i, k in enumerate(kpt):
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf > 0.5:
                            kps.append([int(x_coord), int(y_coord)])
    return kps
