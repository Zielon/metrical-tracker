# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import cv2
import numpy as np
import torch


def get_bbox(image, lmks, bb_scale=2.0):
    h, w, c = image.shape
    lmks = lmks.astype(np.int32)
    x_min, x_max, y_min, y_max = np.min(lmks[:, 0]), np.max(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 1])
    x_center, y_center = int((x_max + x_min) / 2.0), int((y_max + y_min) / 2.0)
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
    xb_min, xb_max, yb_min, yb_max = max(x_center - size // 2, 0), min(x_center + size // 2, w - 1), \
        max(y_center - size // 2, 0), min(y_center + size // 2, h - 1)

    yb_max = min(yb_max, h - 1)
    xb_max = min(xb_max, w - 1)
    yb_min = max(yb_min, 0)
    xb_min = max(xb_min, 0)

    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1

    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])


def crop_image(image, x_min, y_min, x_max, y_max):
    return image[max(y_min, 0):min(y_max, image.shape[0] - 1), max(x_min, 0):min(x_max, image.shape[1] - 1), :]


def squarefiy(image, size=512):
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0)
        image_tensor = input_image.data
    else:
        return input_image.reshape(3, 512, 512).transpose()
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def crop_image_bbox(image, bbox):
    xb_min = bbox[0]
    xb_max = bbox[1]
    yb_min = bbox[2]
    yb_max = bbox[3]
    cropped = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    return cropped
