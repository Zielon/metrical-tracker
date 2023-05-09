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

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import trimesh
from trimesh import Trimesh


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class Masking(nn.Module):
    def __init__(self):
        dir = os.path.abspath(os.path.dirname(__file__))
        super(Masking, self).__init__()
        with open(f'{dir}/data/FLAME2020/FLAME_masks.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            self.masks = Struct(**ss)

        with open(f'{dir}/data/FLAME2020/generic_model.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.color_mesh = trimesh.load(f'{dir}/data/head_template_color.obj', process=False)
        self.color_mask = (np.array(self.color_mesh.visual.vertex_colors[:, 0:3]) == [255, 0, 0])[:, 0].nonzero()[0]
        self.color_mask = np.array([i for i in self.color_mask if i not in self.get_mask_eyes()])

        self.dtype = torch.float32
        self.register_buffer('faces', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('vertices', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

    def to_render_mask(self, mask):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, mask, :] = 1.0
        return face_mask

    def get_faces(self):
        return self.faces

    def get_color_mask(self):
        return self.color_mask

    def get_mask_face(self):
        return self.masks.face

    def get_mask_lips(self):
        return self.masks.lips

    def get_mask_rendering(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.masks.face, :] = 1.0
        face_mask[:, self.masks.left_eyeball, :] = 1.0
        face_mask[:, self.masks.right_eyeball, :] = 1.0

        # face_mask = torch.ones_like(self.vertices)[None]
        # face_mask[:, self.masks.boundary, :] = 0.0
        # face_mask[:, self.masks.left_ear, :] = 0.0
        # face_mask[:, self.masks.right_ear, :] = 0.0
        return face_mask

    def get_mask_depth(self):
        face_mask = torch.ones_like(self.vertices)[None]
        face_mask[:, self.masks.boundary, :] = 0.0
        face_mask[:, self.masks.left_ear, :] = 0.0
        face_mask[:, self.masks.right_ear, :] = 0.0
        return face_mask

    def get_mask_eyes(self):
        left = self.masks.left_eyeball
        right = self.masks.right_eyeball

        return np.unique(np.concatenate((left, right)))

    def get_mask_eyes_rendering(self):
        eyes_mask = torch.zeros_like(self.vertices)[None]
        eyes_mask[:, self.get_mask_eyes(), :] = 1.0

        return eyes_mask

    def get_mask_eyes_region(self):
        left = self.masks.left_eye_region
        right = self.masks.right_eye_region
        mask = np.unique(np.concatenate((left, right)))

        return mask

    def get_mask_eyes_region_rendering(self):
        left = self.masks.left_eye_region
        right = self.masks.right_eye_region

        mask = np.unique(np.concatenate((left, right)))
        eyes_mask = torch.zeros_like(self.vertices)[None]
        eyes_mask[:, mask, :] = 1.0

        return eyes_mask

    def get_mask_ears(self):
        left = self.masks.left_ear
        right = self.masks.right_ear

        return np.unique(np.concatenate((left, right)))

    def get_triangle_face_mask(self):
        m = self.color_mask
        return self.get_triangle_mask(m)

    def get_triangle_color_face_mask(self):
        m = self.masks.face
        return self.get_triangle_mask(m)

    def get_triangle_eyes_mask(self):
        m = self.get_mask_eyes()
        return self.get_triangle_mask(m)

    def get_triangle_whole_mask(self):
        m = self.get_whole_mask()
        return self.get_triangle_mask(m)

    def get_triangle_mask(self, m):
        f = self.faces.cpu().numpy()
        selected = []
        for i in range(f.shape[0]):
            l = f[i]
            valid = 0
            for j in range(3):
                if l[j] in m:
                    valid += 1
            if valid == 3:
                selected.append(i)

        return np.unique(selected)

    def get_binary_triangle_mask(self):
        mask = self.get_whole_mask()
        faces = self.faces.cpu().numpy()
        reduced_faces = []
        for f in faces:
            valid = 0
            for v in f:
                if v in mask:
                    valid += 1
            reduced_faces.append(True if valid == 3 else False)

        return reduced_faces

    def get_masked_faces(self):
        if self.masked_faces is None:
            faces = self.faces.cpu().numpy()
            vertices = self.vertices.cpu().numpy()
            m = Trimesh(vertices=vertices, faces=faces, process=False)
            m.update_faces(self.get_binary_triangle_mask())
            self.masked_faces = torch.from_numpy(np.array(m.faces)).cuda().long()[None]

        return self.masked_faces

    def get_masked_mesh(self, vertices, triangle_mask):
        if len(vertices.shape) == 2:
            vertices = vertices[None]
        B, N, V = vertices.shape
        faces = self.faces.cpu().numpy()
        masked_vertices = torch.empty(0, 0, 3).cuda()
        masked_faces = torch.empty(0, 0, 3).cuda()
        for i in range(B):
            m = Trimesh(vertices=vertices[i].detach().cpu().numpy(), faces=faces, process=False)
            m.update_faces(triangle_mask)
            m.process()
            f = torch.from_numpy(np.array(m.faces)).cuda()[None]
            v = torch.from_numpy(np.array(m.vertices)).cuda()[None].float()
            if masked_vertices.shape[1] != v.shape[1]:
                masked_vertices = torch.empty(0, v.shape[1], 3).cuda()
            if masked_faces.shape[1] != f.shape[1]:
                masked_faces = torch.empty(0, f.shape[1], 3).cuda()
            masked_vertices = torch.cat([masked_vertices, v])
            masked_faces = torch.cat([masked_faces, f])

        return masked_vertices, masked_faces
