from glob import glob
from pathlib import Path

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from loguru import logger
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, config):
        source = Path(config.actor)
        self.images = []
        self.device = 'cuda:0'
        self.source = source
        self.config = config
        self.initialize()

    def initialize(self):
        path = Path(self.source, 'images')
        self.images = sorted(glob(f'{str(path)}/*.jpg') + glob(f'{str(path)}/*.png'))

        if self.config.end_frames == 0:
            self.images = self.images[self.config.begin_frames:]

        elif self.config.end_frames != 0:
            self.images = self.images[self.config.begin_frames:-self.config.end_frames]

        logger.info(f'[ImagesDataset] Initialized with {len(self.images)} frames...')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagepath = self.images[index]
        pil_image = Image.open(imagepath).convert("RGB")
        image = F.to_tensor(pil_image)
        shape = None

        shape_path = Path(self.source, 'identity.npy')
        if shape_path.exists():
            shape = np.load(shape_path)
        else:
            logger.error('[ImagesDataset] Shape (identity.npy) not found! Run MICA shape predictor from https://github.com/Zielon/MICA')
            exit(-1)

        lmk_path = imagepath.replace('images', 'kpt').replace('.png', '.npy').replace('.jpg', '.npy')
        lmk_path_dense = imagepath.replace('images', 'kpt_dense').replace('.png', '.npy').replace('.jpg', '.npy')

        lmk = np.load(lmk_path, allow_pickle=True)
        dense_lmk = np.load(lmk_path_dense, allow_pickle=True)

        lmks = torch.from_numpy(lmk).float()
        dense_lmks = torch.from_numpy(dense_lmk).float()
        shapes = torch.from_numpy(shape).float()

        payload = {
            'image': image,
            'lmk': lmks,
            'dense_lmk': dense_lmks,
            'shape': shapes
        }

        return payload
