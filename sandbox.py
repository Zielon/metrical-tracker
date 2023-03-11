import os
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotx
import numpy as np
import torch
import trimesh

from cv2 import imread, imwrite
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from flame.FLAME import FLAME
from util import dict2obj
import torch.nn as nn


def get_flame():
    mount = '' if os.path.exists('/is/rg/ncs/projects/wzielonka/') else '/home/wzielonka/Cluster'
    config = {
        # FLAME
        'flame_geom_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/generic_model.pkl',
        'flame_template_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/uv_template.obj',
        'flame_static_lmk_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/flame_static_embedding_68_v4.npz',
        'flame_dynamic_lmk_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/flame_dynamic_embedding.npy',
        'mediapipe_lmk_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/mediapipe_landmark_embedding.npz',
        'tex_space_path': f'{mount}/is/rg/ncs/projects/wzielonka/tracker/data/FLAME_albedo_from_BFM.npz',
        'dtype': torch.float32,
        'num_exp_params': 100,
        'num_shape_params': 300,
        'tex_params': 150,
    }

    config = dict2obj(config)
    flame = FLAME(config).cuda()
    return flame


def generate_meshes(source, faces):
    shape = source.replace('results_dict.npy', '00000.frame')
    flame = torch.load(shape)['flame']
    meshes = np.load(source, allow_pickle=True)[0]['prediction_dict']['justin_long_only_exprs_condition_justin_long_only_exprs']
    size = meshes.shape
    meshes = meshes.reshape(size[0], int(size[1] / 3), 3)
    results = []
    for vertices in meshes:
        tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        results.append(tri)

    return results, flame


def dump_text(path, params):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, params, fmt='%.8f')


def get_expressions(meshes, face_model, frame):
    n_vertices = face_model.v_template.shape[0]
    n_eigenvectors = 400
    expressions = []
    T = face_model.v_template.cpu().numpy()

    shape = torch.from_numpy(frame['shape']).cuda()

    basis = face_model.shapedirs[:, :, :n_eigenvectors].cpu().numpy()
    basis = np.reshape(basis, (-1, n_eigenvectors))
    scale = np.linalg.norm(basis, axis=0)
    basis_normalized = basis / scale[None, :]
    inv_basis = np.diag(1.0 / scale).dot(basis_normalized.T)

    inv_basis = inv_basis.reshape((n_eigenvectors, n_vertices, 3))
    inv_basis = inv_basis.reshape((n_eigenvectors, -1))

    # betas = inv_basis.dot((v - T).reshape(-1, ))

    for j, mesh in tqdm(enumerate(meshes)):
        exp = optimize_expressions(flame, shape, torch.from_numpy(mesh.vertices).cuda())
        test = face_model(shape_params=shape, expression_params=exp)[0][0].cpu().numpy()
        i = str(j).zfill(5)
        dump_text(f'/home/wzielonka/PycharmProjects/smplx-internal/inha/impregnator/flame/exp/{i}.txt', exp.detach().cpu().numpy()[0])
        trimesh.Trimesh(vertices=test, faces=mesh.faces, process=False).export(f'/home/wzielonka/PycharmProjects/smplx-internal/inha/impregnator/meshes/{i}.ply')
        # trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).export('test_b.obj')

    return expressions


def optimize_expressions(flame, shape, target):
    exp = nn.Parameter(torch.zeros([1, 100]).cuda())
    params = [{'params': [exp], 'lr': 0.1}]

    optimizer = torch.optim.Adam(params)

    for i in range(100):
        vertices = flame(shape_params=shape, expression_params=exp)[0]

        loss = (target - vertices).abs().mean()
        # loss += torch.sum(exp ** 2) * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return exp.detach()


if __name__ == '__main__':
    # flame = get_flame()
    # faces = flame.faces.cpu().numpy()
    # source = '/home/wzielonka/Cluster/is/rg/ncs/projects/bala/datasets/justin_christmas/results_dict.npy'
    # meshes, frame = generate_meshes(source, faces)
    # get_expressions(meshes, flame, frame)

    # for actor in ['jalees_1', 'wojtek_1', 'malte_1', 'obama', 'biden', 'justin', 'justin_1']:
    #     for mode in ['test', 'train']:
    #         root = f'/home/wzielonka/Cluster/is/rg/ncs/projects/wzielonka/inha/imavatar/{actor}/{actor}/{mode}/orig_image'
    #         dst = root.replace('orig_image', 'image')
    #         if not os.path.exists(root):
    #             os.system(f'mv {dst} {root}')
    #         Path(root.replace('orig_image', 'image')).mkdir(parents=True, exist_ok=True)
    #         dst = root.replace('orig_image', 'mask')
    #         dst2 = root.replace('orig_image', 'orig_mask')
    #         if not os.path.exists(dst2):
    #             os.system(f'mv {dst} {dst2}')
    #         Path(root.replace('orig_image', 'mask')).mkdir(parents=True, exist_ok=True)
    #         for path in tqdm(sorted(glob(root + "/*.png"))):
    #             img = cv2.imread(path, cv2.IMREAD_UNCHANGED) / 255.0
    #             matting = cv2.imread(path.replace('orig_image', 'orig_mask'), cv2.IMREAD_UNCHANGED)[:, :, 0:1] / 255.0
    #             img *= matting
    #             img += (1 - matting)
    #             img = (img * 255).astype(np.uint8)
    #             mask = cv2.imread(path.replace('orig_image', 'semantic'), cv2.IMREAD_UNCHANGED)
    #             mask = (mask == 16).astype(np.int32)[:, :, None]  # remove clothes
    #             mask = ndimage.median_filter(mask, size=5)
    #             mask = (ndimage.binary_dilation(mask, iterations=1) > 0).astype(np.uint8)
    #             img[:, :, 0:3] *= (1 - mask)
    #             matting[:, :, 0:3] *= (1 - mask)
    #             img[:, :, 0:3] += np.ones_like(mask) * 255
    #             alpha = img[:, :, 3:4]
    #             alpha = np.where(mask, np.zeros_like(alpha), alpha)
    #             img[:, :, 3:4] = alpha
    #
    #             cv2.imwrite(path.replace('orig_image', 'image'), img)
    #             cv2.imwrite(path.replace('orig_image', 'mask'), matting * 255)

    # dst = Path("/home/wzielonka/Downloads/Neural_Head_Avatars_Supplemental/ckpts_and_data/data/philip")
    # dst.mkdir(parents=True, exist_ok=True)
    # for path in tqdm(sorted(glob("/home/wzielonka/Downloads/Neural_Head_Avatars_Supplemental/ckpts_and_data/data/person_0000/frame_*"))):
    #     name = Path(path).stem.split('_')[-1]
    #     src = "image_0000.png"
    #     os.system(f'cp {path}/{src} {str(dst)}/{name}.png')

    # kids = imread('/home/wzielonka/PycharmProjects/tracker-internal/kids.png')
    # adults = imread('/home/wzielonka/PycharmProjects/tracker-internal/adults.png')
    #
    # kids_feng = []
    # kids_li = []
    # kids_ours = []
    # kids_input = []
    #
    # for i in [0, 2]:
    #     x = 512 * i
    #     y = 0
    #     kids_input.append(kids[y:y+512, x:x+512, :])
    #     y += 512
    #     kids_ours.append(kids[y:y+512, x:x+512, :])
    #     y += 512
    #     kids_li.append(kids[y:y+512, x:x+512, :])
    #     y += 512
    #     kids_feng.append(kids[y:y+512, x:x+512, :])
    #
    # adults_feng = []
    # adults_li = []
    # adults_ours = []
    # adults_input = []
    #
    # for j in [53, 52, 46, 7, 55]:
    #     y = 512 * j
    #     x = 0
    #     adults_input.append(adults[y:y+512, x:x+512, :])
    #     x = 512 * 2
    #     adults_li.append(adults[y:y+512, x:x+512, :])
    #     x = 512 * 4
    #     adults_feng.append(adults[y:y+512, x:x+512, :])
    #     x = 512 * 5
    #     adults_ours.append(adults[y:y+512, x:x+512, :])
    #
    # feng = kids_feng + adults_feng
    # li = kids_li + adults_li
    # ours = kids_ours + adults_ours
    # input = kids_input + adults_input
    #
    # columns = []
    # for i in range(len(feng)):
    #     column = np.concatenate([
    #         input[i],
    #         ours[i],
    #         li[i],
    #         feng[i]
    #     ], axis=0)
    #     columns.append(column)
    #
    # img = np.concatenate(columns, axis=1)
    # imwrite('output.png', img)

    # src = '/home/wzielonka/datasets/inha/videos/'
    # dst = '/home/wzielonka/datasets/inha/dataset/'
    # actors = ['justin_1', 'justin']
    #
    # for actor in tqdm(actors):
    #     video = src + actor + '.mp4'
    #     tmp = Path('tmp')
    #     tmp.mkdir(parents=True, exist_ok=True)
    #     os.system(f'ffmpeg -i {video} -vf fps=25 -q:v 1 tmp/%05d.png')
    #     output = dst + actor
    #     Path(output).mkdir(parents=True, exist_ok=True)
    #     images = sorted(glob('tmp/*.png'))
    #     for img in images[:-350]:
    #         os.system(f'cp {img} {output}')
    #     os.system(f'ffmpeg -y -framerate 25 -pattern_type glob -i \'{output}/*.png\' -c:v libx264 -crf 16 {output}/train.mp4')
    #     os.system(f'rm -rf {output}/*.png')
    #     for img in images[-350:]:
    #         os.system(f'cp {img} {output}')
    #     os.system(f'ffmpeg -y -framerate 25 -pattern_type glob -i \'{output}/*.png\' -c:v libx264 -crf 16 {output}/test.mp4')
    #     os.system(f'rm -rf {output}/*.png')
    #     os.system(f'rm -rf tmp/*')

    # from PIL import ImageFont, ImageDraw, Image
    #
    # font = ImageFont.truetype("/home/wzielonka/Downloads/times.ttf", 50)
    #
    # def draw_text(img, text, font, color, offset_text):
    #     cv2_im_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     pil_im = Image.fromarray(cv2_im_rgb)
    #     draw = ImageDraw.Draw(pil_im)
    #     draw.text((offset_text + 5, 70), text, font=font, fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    #     img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    #     return img
    #
    # def color(word, offset_frame, offset_text, img, color=None, black_image=None):
    #     fps = 30
    #     text, duration, _ = word
    #     if color is None:
    #         color = word[2]
    #     num_frames = int(duration * fps)
    #     shift = font.getsize(text)[0] + font.getsize(" ")[0]
    #
    #     if black_image is not None:
    #         img[:, :offset_text + shift, :] = black_image[:, :offset_text + shift, :]
    #     img[:, offset_text:offset_text + shift, :] = 255
    #
    #     img = draw_text(img, text, font, color, offset_text)
    #
    #     for i in range(num_frames):
    #         cv2.imwrite(f'color/{str(offset_frame + i).zfill(3)}.png', img)
    #
    #     offset_frame += num_frames
    #     offset_text += shift
    #     return offset_frame, offset_text, img
    #
    # img = np.ones([200, 1000, 3]) * 255.0
    # sentence = [("My", 0.5, (1, 0, 0)), ("name", 1, (1, 0, 0)), ("is", 0.5, (1, 0, 0)), ("Bala", 1, (1, 0, 0)), ("Bala", 1, (0.5, 0.5, 0.5)), ("the", 1, (0.5, 0.5, 0.5)), ("impregnator", 1, (1, 0, 0))]
    #
    # os.system(f'rm -rf color/*')
    #
    # offset_frame = 0
    # offset_text = 0
    # for word in sentence:
    #     offset_frame, offset_text, img = color(word, offset_frame, offset_text, img, (0, 0, 0))
    #
    # black_image = img.copy()
    # offset_frame = 0
    # offset_text = 0
    # for word in sentence:
    #     offset_frame, offset_text, img = color(word, offset_frame, offset_text, img, black_image=black_image)
    #
    # os.system(f'ffmpeg -y -framerate 30 -pattern_type glob -i \'color/*.png\' -c:v libx264 -crf 1 color.mp4')

    out_file = "/home/wzielonka/Cluster/is/rg/ncs/projects/bala/live_graph/0024.npy"
    all_rows = np.load(out_file, allow_pickle=True)
    title_font = {'fontname': 'DejaVu Sans', 'size': '12', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    def job(seq):
        experiment_id = seq.split('.')[0]
        dst = Path('animations', experiment_id)
        dst.mkdir(parents=True, exist_ok=True)
        # colors_map = iter(cm.gist_rainbow(np.linspace(0, 1, 5)))
        with plt.style.context(matplotx.styles.dufte):
            gt_verts = all_rows[0]["lip_dict"][seq]["GT"]
            for j in (range(gt_verts.shape[0] - 1)):
                j += 1
                fig = plt.figure(figsize=(15, 6))
                empty = np.array([np.nan for i in range(gt_verts.shape[0])])
                colors_map = iter([plt.cm.Set1(i) for i in range(0, 5)])
                x = np.arange(gt_verts.shape[0])

                plt.xticks(np.arange(0, gt_verts.shape[0], 10))

                plt.ylim(0, np.max(gt_verts) + 1)
                plt.xlim(0, gt_verts.shape[0] + 1)

                c = next(colors_map)
                plt.plot(x, np.concatenate([gt_verts[:j], empty[j:]], axis=0), color=c, label="GT")

                for row in all_rows:

                    current_dict = row["lip_dict"].get(seq, None)
                    if current_dict is None:
                        continue

                    model_name = row["model_name"]
                    for condition, lip_dist in current_dict.items():
                        if condition is "GT":
                            continue
                        colour = next(colors_map)
                        plt.plot(x, np.concatenate([lip_dist[:j], empty[j:]], axis=0), color=colour, label=model_name)

                plt.legend(loc="upper right")
                plt.title("Lip Distance Curve", **title_font)
                plt.tight_layout()
                fig.savefig(f'{str(dst)}/{str(j).zfill(3)}.png', dpi=500)
                plt.close(fig)

        os.system(f'ffmpeg -y -framerate 30 -pattern_type glob -i \'animations/{experiment_id}/*.png\' -c:v libx264 -crf 1 animations/{experiment_id}.mp4')


    torch.multiprocessing.set_start_method('spawn')
    with torch.multiprocessing.Pool(processes=8) as pool:
        sequences = list(all_rows[0]["lip_dict"].keys())
        for task in tqdm(pool.imap_unordered(job, sequences), total=len(sequences)):
            pass
