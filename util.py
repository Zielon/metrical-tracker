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

import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from flame.mediapipe.landmarks import LIPS_LANDMARK_IDS, get_idx, NOSE_LANDMARK_IDS

l1_loss = nn.SmoothL1Loss(beta=0.1)

face_mask = torch.ones([1, 68, 2]).cuda().float()
nose_mask = torch.ones([1, 68, 2]).cuda().float()
oval_mask = torch.ones([1, 68, 2]).cuda().float()

face_mask[:, [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], :] = 0
nose_mask[:, [27, 28, 29, 30, 31, 32, 33, 34, 35], :] *= 4.0
oval_mask[:, [i for i in range(17)], :] *= 0.4

nose_mask_mp = torch.ones([1, 105, 2]).cuda().float()
face_mask_mp = torch.ones([1, 105, 2]).cuda().float()

nose_mask_mp[:, get_idx(NOSE_LANDMARK_IDS), :] *= 8.0


# face_mask_mp[:, get_idx(LEFT_EYE_LANDMARK_IDS) + get_idx(RIGHT_EYE_LANDMARK_IDS), :] *= 0.1


# Input is R, t in opencv spave
def opencv_to_opengl(R, t):
    # opencv is row major
    # opengl is column major
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t

    Rt[[1, 2]] *= -1  # opencv to opengl coordinate system swap y,z

    '''
            | R | t |
            | 0 | 1 |

            inverse is

            | R^T | -R^T * t |
            | 0   | 1        |

    '''

    # Transpose rotation (row to column wise) and adjust camera position for the new rotation matrix
    Rt = np.linalg.inv(Rt)
    return Rt


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


def scale_lmks(opt_lmks, target_lmks, image_size):
    h, w = image_size
    size = torch.tensor([1 / w, 1 / h]).float().cuda()[None, None, ...]
    opt_lmks = opt_lmks * size
    target_lmks = target_lmks * size
    return opt_lmks, target_lmks


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask):
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff = torch.pow(opt_lmks - target_lmks, 2)
    return (diff * lmk_mask).mean()


def face_lmk_loss(opt_lmks, target_lmks, image_size, is_mediapipe, lmk_mask):
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if not is_mediapipe:
        return (diff * face_mask * nose_mask * oval_mask * lmk_mask).mean()
    return (diff * nose_mask_mp * lmk_mask).mean()


def oval_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask):
    oval_ids = [i for i in range(17)]
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff = torch.pow(opt_lmks[:, oval_ids, :] - target_lmks[:, oval_ids, :], 2)
    return (diff * lmk_mask[:, oval_ids, :]).mean()


def mouth_lmk_loss(opt_lmks, target_lmks, image_size, is_mediapipe, lmk_mask):
    if not is_mediapipe:
        mouth_ids = [i for i in range(49, 68)]
    else:
        mouth_ids = get_idx(LIPS_LANDMARK_IDS)
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff = torch.pow(opt_lmks[:, mouth_ids, :] - target_lmks[:, mouth_ids, :], 2)
    return (diff * lmk_mask[:, mouth_ids, :]).mean()


def eye_closure_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask):
    upper_eyelid_lmk_ids = [47, 46, 45, 29, 30, 31]
    lower_eyelid_lmk_ids = [39, 40, 41, 25, 24, 23]
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff_opt = opt_lmks[:, upper_eyelid_lmk_ids, :] - opt_lmks[:, lower_eyelid_lmk_ids, :]
    diff_target = target_lmks[:, upper_eyelid_lmk_ids, :] - target_lmks[:, lower_eyelid_lmk_ids, :]
    diff = torch.pow(diff_opt - diff_target, 2)
    return (diff * lmk_mask[:, upper_eyelid_lmk_ids, :]).mean()


def mouth_closure_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask):
    upper_mouth_lmk_ids = [49, 50, 51, 52, 53, 61, 62, 63]
    lower_mouth_lmk_ids = [59, 58, 57, 56, 55, 67, 66, 65]
    opt_lmks, target_lmks = scale_lmks(opt_lmks, target_lmks, image_size)
    diff_opt = opt_lmks[:, upper_mouth_lmk_ids, :] - opt_lmks[:, lower_mouth_lmk_ids, :]
    diff_target = target_lmks[:, upper_mouth_lmk_ids, :] - target_lmks[:, lower_mouth_lmk_ids, :]
    diff = torch.pow(diff_opt - diff_target, 2)
    return (diff * lmk_mask[:, upper_mouth_lmk_ids, :]).mean()


def pixel_loss(opt_img, target_img, mask=None):
    if mask is None:
        mask = torch.ones_like(opt_img)
    n_pixels = torch.sum((mask[:, 0, ...] > 0).int()).detach().float()
    loss = (mask * (opt_img - target_img)).abs()
    loss = torch.sum(loss) / n_pixels
    return loss


def reg_loss(params):
    return torch.mean(torch.sum(torch.square(params), dim=1))


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def tensor_vis_landmarks(images, landmarks, color='g'):
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()

    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = (image * 255)
        predicted_landmark = predicted_landmarks[i]
        image_landmarks = plot_all_kpts(image, predicted_landmark, color)
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpts(image, kpts, color='r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    c = (0, 100, 255)
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)

    image = image.copy()
    kpts = kpts.copy()

    # for j in range(kpts.shape[0] - 17):
    for j in range(kpts.shape[0]):
        # i = j + 17
        st = kpts[j, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)
        if j in end_list:
            continue
        ed = kpts[j + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), 1)

    return image


def plot_all_kpts(image, kpts, color='b'):
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    elif color == 'p':
        c = (255, 100, 100)

    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)

    return image


def get_gaussian_pyramid(levels, input, kernel_size, sigma):
    pyramid = []
    images = input.clone()
    for k, level in enumerate(reversed(levels)):
        image_size, iters = level
        size = [int(image_size[0]), int(image_size[1])]
        images = F.interpolate(images, size, mode='bilinear', align_corners=False)
        images = gaussian_blur(images, [kernel_size, kernel_size], sigma=[sigma, sigma] if sigma is not None else None)
        pyramid.append((images, iters, size, image_size))

    return list(reversed(pyramid))


def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w - 1 - margin_x):
        for y in range(margin_y, h - 1 - margin_y):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


def get_aspect_ratio(images):
    h, w = images.shape[2:4]
    ratio = w / h
    if ratio > 1.0:
        aspect_ratio = torch.tensor([1. / ratio, 1.0]).float().cuda()[None]
    else:
        aspect_ratio = torch.tensor([1.0, ratio]).float().cuda()[None]
    return aspect_ratio


def is_optimizable(name, param_groups):
    for param in param_groups:
        if name.strip() in param['name']:
            return True
    return False


def merge_views(views):
    grid = []
    for view in views:
        grid.append(np.concatenate(view, axis=2))
    grid = np.concatenate(grid, axis=1)

    # tonemapping
    return to_image(grid)


def to_image(img):
    img = (img.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    return img


def dump_point_cloud(name, view):
    _, _, h, w = view.shape
    np.savetxt(f'pc_{name}.xyz', view.permute(0, 2, 3, 1).reshape(h * w, 3).detach().cpu().numpy(), fmt='%f')


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def images_to_video(path, fps=25, src='video', video_format='DIVX'):
    img_array = []
    for filename in tqdm(sorted(glob.glob(f'{path}/{src}/*.jpg'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        out = cv2.VideoWriter(f'{path}/video.avi', cv2.VideoWriter_fourcc(*video_format), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def grid_sample(image, optical, align_corners=False):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def get_flame_extra_faces():
    return torch.from_numpy(
        np.array(
            [[1573, 1572, 1860],
             [1742, 1862, 1572],
             [1830, 1739, 1665],
             [2857, 2862, 2730],
             [2708, 2857, 2730],
             [1862, 1742, 1739],
             [1830, 1862, 1739],
             [1852, 1835, 1666],
             [1835, 1665, 1666],
             [2862, 2861, 2731],
             [1747, 1742, 1594],
             [3497, 1852, 3514],
             [1595, 1747, 1594],
             [1746, 1747, 1595],
             [1742, 1572, 1594],
             [2941, 3514, 2783],
             [2708, 2945, 2857],
             [2941, 3497, 3514],
             [1852, 1666, 3514],
             [2930, 2933, 2782],
             [2933, 2941, 2783],
             [2862, 2731, 2730],
             [2945, 2930, 2854],
             [1835, 1830, 1665],
             [2857, 2945, 2854],
             [1572, 1862, 1860],
             [2854, 2930, 2782],
             [2708, 2709, 2943],
             [2782, 2933, 2783],
             [2708, 2943, 2945]])).cuda()[None, ...]
