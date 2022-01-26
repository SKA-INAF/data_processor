from genericpath import exists
import random, json, os, math
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import cv2
from pathlib import Path
from torchvision.utils import save_image
from abc import ABC, abstractmethod

from astropy.visualization import ZScaleInterval

class NpEncoder(json.JSONEncoder):
    # JSON Encoder class to manage output file saving
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class DefaultParser(ABC):

    def __init__(self, dst_dir, split):
        self.CLASSES = {'sidelobe': 1, 'source': 2, 'galaxy': 3}
        self.dst_dir = dst_dir
        self.split = split
        self.json_enc = NpEncoder
        self.error_file = Path(f'{self.split}_skipped.txt')
        if self.error_file.exists():
            self.error_file.unlink()

    def read_samples(self, trainset_path):
        '''trainset.dat file parsing to get dataset samples'''
        samples = []
        with open(trainset_path) as f:
            for json_path in tqdm(f):
                json_rel_path = Path(json_path.strip())
                abs_json_path = trainset_path.parent / json_rel_path
                with open(abs_json_path, 'r') as label_json:
                    label = json.load(label_json)
                    # replacing relative path with the absolute one
                    label['img'] = trainset_path.parent / Path('imgs') / label['img']
                    samples.append(label)

            return samples

    def train_val_split(self, samples, val_ratio=0.1, test_ratio=0.005):
        '''trainset.dat file parsing to get a random train-val split'''

        random.shuffle(samples)
        test_sep = math.floor(len(samples) * test_ratio)
        val_sep = math.floor(len(samples) * val_ratio)
        assert (len(samples) / test_sep + val_sep) > 0.8, f'Less than 80% of data for training'
        val_entries = samples[ : val_sep]
        test_entries = samples[val_sep : val_sep + test_sep ]
        train_entries = samples[val_sep + test_sep : ]

        return train_entries, val_entries, test_entries

    @abstractmethod
    def make_img_dir(entries, split):
        '''Copies images into train or val folder'''
        return

    def log_error(self, msg):
        with open(self.dst_dir / self.error_file, 'a') as td:
            td.write(msg + '\n')


    def fits_to_png(self, file_path, dst_path, contrast=0.15):
        
        img = fits.getdata(file_path, ignore_missing_end=True)
        interval = ZScaleInterval(contrast=contrast)
        min, max = interval.get_limits(img)

        img = (img-min)/(max-min)

        save_image(torch.from_numpy(img), dst_path)

    def get_mask_coords(self, mask_path):
        '''Extracts coordinates from the mask image'''
        img = fits.getdata(mask_path).astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x_points = []
        y_points = []

        if not contours:
            return None, None

        for xy in contours[0]:
            x_points.append(xy[0][0])
            y_points.append(xy[0][1])
        return x_points, y_points
    
    def get_img_size(self, img_path):
        '''Extracts size from the mask image'''
        img = fits.getdata(img_path).astype(np.uint8)
        return img.shape

    @abstractmethod
    def make_annotations(self, samples, split, incremental_id):
        '''Creates the JSON COCO annotations to be stored'''
        return