import argparse
import os
from pathlib import Path

from yacs.config import CfgNode as CN

cfg = CN()

local = os.path.exists("/home/wzielonka/Cluster") or os.path.exists("/home/wzielonka-local/Cluster")

cfg.flame_geom_path = 'data/FLAME2020/generic_model.pkl'
cfg.flame_template_path = 'data/uv_template.obj'
cfg.flame_lmk_path = 'data/FLAME2020/landmark_embedding.npy'
cfg.tex_space_path = 'data/FLAME2020/FLAME_albedo_from_BFM.npz'

cfg.num_shape_params = 300
cfg.num_exp_params = 100
cfg.tex_params = 140
cfg.actor = ''
cfg.config_name = ''
cfg.kernel_size = 7
cfg.sigma = 9.0
cfg.keyframes = [0]
cfg.bbox_scale = 2.5
cfg.fps = 25
cfg.begin_frames = 0
cfg.end_frames = 0
cfg.image_size = [512, 512]  # height, width
cfg.rotation_lr = 0.01
cfg.translation_lr = 0.003
cfg.sampling = [[0.5, 90], [1.0, 80], [2.0, 70]]
cfg.optimize_shape = False
cfg.crop_image = True

cfg.save_folder = './test_results/'

# Weights
cfg.w_pho = 350
cfg.w_lmks = 100
cfg.w_lmks_lid = 20
cfg.w_lmks_mouth = 80
cfg.w_lmks_iris = 4
cfg.w_lmks_oval = 30

cfg.w_exp = 0.02
cfg.w_shape = 0.5
cfg.w_tex = 0.04


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Configuration file', required=True)

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    cfg.config_name = Path(args.cfg).stem

    if local:
        cfg.save_folder = './test_results/'

    if not local:
        cfg.actor = cfg.actor.replace('PycharmProjects', 'projects')

    return cfg


def parse_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg.cfg_file = cfg_file

    cfg.config_name = Path(cfg_file).stem

    return cfg
