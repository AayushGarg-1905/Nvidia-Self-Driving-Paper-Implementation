import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class NvidiaDataset(Dataset):
    def __init__(self, dataframe, image_dir, is_training=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        choice = random.choice(['center', 'left', 'right']) if self.is_training else 'center'
        img_path = os.path.join(self.image_dir, os.path.basename(row[choice].strip()))
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        steering = row['steering']
        correction = 0.2

        if choice == 'left':
            steering += correction
        elif choice == 'right':
            steering -= correction

        if self.is_training:
            image, steering = self.apply_augmentations(image, steering)

        image = self.preprocess(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        steering = torch.tensor([steering], dtype=torch.float32)

        return image, steering

    def apply_augmentations(self, image, steering):

        if random.random() < 0.8:
            tx = np.random.randint(-50, 51)
            steering += tx * 0.002
            trans_matrix = np.float32([[1, 0, tx], [0, 1, 0]])
            image = cv2.warpAffine(image, trans_matrix, (image.shape[1], image.shape[0]))

        if random.random() < 0.8:
            image = image.astype(np.float32)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            image[:, :, 0] = np.clip(image[:, :, 0] * ratio, 0, 255) 
            image = image.astype(np.uint8)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            steering = -steering

        return image, steering

    def preprocess(self, image):

        image = image[60:-25, :, :]
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0

        return image
