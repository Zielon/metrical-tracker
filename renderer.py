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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from skimage.io import imread

import util
from masking import Masking
from tracker_rasterizer import TrackerRasterizer

sky = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda()


def apply_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)


def remove_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma) + torch.min(rgb, rgb.new_tensor(0.0))
        return res


class Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=512, flip=False):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]

        mask = torch.from_numpy(imread('data/uv_mask_eyes.png') / 255.).permute(2, 0, 1).cuda()[0:3, :, :]
        mask = mask > 0.
        mask = F.interpolate(mask[None].float(), [2048, 2048], mode='bilinear')

        self.register_buffer('mask', mask)

        self.rasterizer = TrackerRasterizer(image_size, None)
        self.masking = Masking()

        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## lighting
        pi = np.pi
        sh_const = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.register_buffer('constant_factor', sh_const)

    def set_size(self, size):
        self.rasterizer.raster_settings.image_size = size

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ], 1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def render_depth(self, vertices_world, cameras, faces=None):
        self.rasterizer.reset()
        B = vertices_world.shape[0]
        if faces is None:
            faces = self.faces.expand(B, -1, -1)
        meshes_world = Meshes(verts=vertices_world.float(), faces=faces.long())
        face_vertices_view = util.face_vertices(cameras.get_world_to_view_transform().transform_points(vertices_world), faces)
        depth_mask = util.face_vertices(self.masking.get_mask_depth(), faces)
        attributes = torch.cat([face_vertices_view, depth_mask], -1)
        rendering = self.rasterizer(meshes_world, attributes, cameras=cameras)[0]
        view_vertices_images = rendering[:, 0:3, :, :].detach()
        mask = rendering[:, 3:6, :, :].detach() > 0

        return view_vertices_images * mask

    def forward(self, vertices_world, albedos, lights, cameras):
        B = vertices_world.shape[0]
        faces = self.faces.expand(B, -1, -1)

        meshes_world = Meshes(verts=vertices_world.float(), faces=faces.long())
        meshes_ndc = self.rasterizer.transform(meshes_world, cameras=cameras)
        vertices_ndc = meshes_ndc.verts_padded()

        face_mask = util.face_vertices(self.masking.to_render_mask(self.masking.get_mask_face()), faces)
        render_mask = util.face_vertices(self.masking.get_mask_rendering(), faces)
        depth_mask = util.face_vertices(self.masking.get_mask_depth(), faces)
        eyes_region_mask = util.face_vertices(self.masking.get_mask_eyes_region_rendering(), faces)
        eyes_mask = util.face_vertices(self.masking.get_mask_eyes_rendering(), faces)
        face_vertices_ndc = util.face_vertices(vertices_ndc, faces)
        face_vertices_view = util.face_vertices(cameras.get_world_to_view_transform().transform_points(vertices_world), faces)
        face_normals = meshes_world.verts_normals_packed()[meshes_world.faces_packed()][None]
        uv = self.face_uvcoords.expand(B, -1, -1, -1)

        attributes = torch.cat([uv, face_vertices_ndc, face_normals, face_mask, face_vertices_view, render_mask, depth_mask, eyes_region_mask, eyes_mask], -1)
        rendering, zbuffer = self.rasterizer(meshes_world, attributes, cameras=cameras)

        uvcoords_images = rendering[:, 0:3, :, :].detach()
        ndc_vertices_images = rendering[:, 3:6, :, :]
        normal_images = rendering[:, 6:9, :, :].detach()
        mask_images_mesh = rendering[:, 9:12, :, :].detach()
        view_vertices_images = rendering[:, 12:15, :, :]
        mask_images_rendering = rendering[:, 15:18, :, :].detach()
        mask_images_depth = rendering[:, 18:21, :, :].detach()
        mask_images_eyes_region = rendering[:, 21:24, :, :].detach()
        mask_images_eyes = rendering[:, 24:27, :, :].detach()
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        mask = self.mask.repeat(B, 1, 1, 1)
        grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False).float()
        mask_images = F.grid_sample(mask, grid, align_corners=False).float()
        shading_images = self.add_SHlight(normal_images, lights)
        images = albedo_images * shading_images

        outputs = {
            'grid': grid,
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'mask_images': (mask_images * alpha_images > 0).float(),
            'mask_images_mesh': (mask_images_mesh > 0).float(),
            'mask_images_rendering': (mask_images_rendering > 0).float(),
            'mask_images_depth': (mask_images_depth > 0).float(),
            'mask_images_eyes_region': (mask_images_eyes_region > 0).float(),
            'mask_images_eyes': (mask_images_eyes > 0).float(),
            'position_images': ndc_vertices_images,
            'position_view_images': view_vertices_images,
            'zbuffer': zbuffer
        }

        return outputs
