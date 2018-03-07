import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset


class StereoDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_classes, means, use_stereo=False):
        self.means = means # mean of three channels in the order of RGB
        self.input_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.stereo = use_stereo
        self.num_c = num_classes

    def __len__(self):
        return len(self.input_csv)

    def __getitem__(self, idx):

        # get left image
        left_image_name = os.path.join(self.root_dir, str(self.input_csv.iloc[idx, 0]))
        left_image = io.imread(left_image_name)
        left_image = self.subtract_mean(left_image.astype(float))

        # get label image
        label_image_name = os.path.join(self.root_dir, str(self.input_csv.iloc[idx, 2]))

        label_image = io.imread(label_image_name)
        label_image = label_image[:, :, 0]
        label_image = torch.from_numpy(label_image.copy()).long()

        # get right image if stereo else return
        if self.stereo:
            right_image_name = os.path.join(self.root_dir, str(self.input_csv.iloc[idx, 1]))

            right_image = io.imread(right_image_name)
            right_image = self.subtract_mean(right_image.astype(float))

            return {'XL': left_image, 'XR': right_image, 'Y': label_image}
        else:
            return {'XL': left_image, 'Y': label_image}

    # TRANSPOSE AND SUBTRACT MEAN
    def subtract_mean(self, img):
        img[:, :, 0] -= self.means[0]
        img[:, :, 1] -= self.means[1]
        img[:, :, 2] -= self.means[2]
        img = np.transpose(img, (2, 0, 1)) / 255.
        return img
