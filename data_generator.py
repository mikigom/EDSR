import os

import numpy as np
from skimage import io


def load_image_from_dir(directory, rank=None):
    file_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(path, name))

    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    images = []
    if rank is None:
        for i in range(len(file_list)):
            images.append(io.imread(file_list[i]).astype(np.float32)/255.)
    else:
        rank_interal = int(len(file_list)/4)
        for i in range(rank*rank_interal, (rank+1)*rank_interal):
            images.append(io.imread(file_list[i]).astype(np.float32)/255.)

    return images, file_list
