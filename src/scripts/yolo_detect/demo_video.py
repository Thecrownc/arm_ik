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
import numpy as np

from nets import nn
from utils import util
from utils.dataset import Dataset
import cv2

input_size = 640

if __name__ == "__main__":
    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)

    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    model = torch.load('./weights/100.pt', map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

    base_output_path = '/home/guozz/Datasets/button_datasets/eval'
    frame_rate = 30  # 帧率

    # 遍历文件夹和帧
    for fold in ['rgb_high', 'rgb_left', 'rgb_right']:
        video_output_path = os.path.join(base_output_path, f"{fold}.avi")

        # 视频写入对象初始化为 None
        video_writer = None

        for i in range(0, 601):
            print(fold,i)
            file = str(i).zfill(4) + '.jpg'
            img_path = os.path.join(base_output_path, fold, file)
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"文件未找到或无法读取: {img_path}")
                continue

            # 初始化视频写入对象
            if video_writer is None:
                frame_height, frame_width = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    video_output_path,
                    cv2.VideoWriter_fourcc(*'XVID'),  # 使用 XVID 编码
                    frame_rate,
                    (frame_width, frame_height)  # 视频分辨率
                )

            image = frame.copy()
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, input_size / shape[0], input_size / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = input_size - pad[0]
            h = input_size - pad[1]
            w = np.mod(w, stride)
            h = np.mod(h, stride)
            w /= 2
            h /= 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

            # Convert HWC to CHW, BGR to RGB
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
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
                #     x1, y1, x2, y2, score, index = box
                #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                for kpt in reversed(kps_output):
                    for i, k in enumerate(kpt):
                        color_k = [int(x) for x in kpt_color[i]]
                        x_coord, y_coord = k[0], k[1]
                        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                            if len(k) == 3:
                                conf = k[2]
                                if conf < 0.5:
                                    continue
                            cv2.circle(frame, (int(x_coord), int(y_coord)), 5, color_k, -1, lineType=cv2.LINE_AA)

            # 写入当前帧到视频
            video_writer.write(frame)

        # 释放视频写入对象
        if video_writer is not None:
            video_writer.release()
            print(f"视频已保存到: {video_output_path}")