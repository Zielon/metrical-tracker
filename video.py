from glob import glob
import os

from tqdm import tqdm


def video():
    for actor in tqdm(sorted(filter(os.path.isdir, glob(f'output/*')))):
        os.system(f'ffmpeg -y -framerate 25 -pattern_type glob -i \'{actor}/video/*.jpg\' -c:v libx264 {actor}.mp4')


if __name__ == '__main__':
    video()
