from pathlib import Path
import random
import argparse
from dataset.utils import *
from tqdm import tqdm
from dataset.galaxy import build as build_galaxy, inv_normalize
import torchvision.transforms.functional as F

from dataset.utils import box_cxcywh_to_xyxy

label_idx_to_name = ['None', "Extended", "Compact", "Spurious"]

def crop_dataset(split):
    dset = build_galaxy(split, data_path, args.masks)
    counter = {'Extended': 0, 'Compact': 0, 'Spurious': 0}

    for sample in tqdm(dset):
        image, target = sample
        boxes = target['boxes']
        masks = target['masks'].float()
        labels = target['labels']
        img_path = Path(target['path'])
        image = image.unsqueeze(0)
        masks = masks.unsqueeze(1)

        H, W = image.shape[2:]

        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes_to_pixel_coords(boxes, W, H)

        norm_image = inv_normalize(image)
        crops = norm_image * masks

        for i, (box, label, crop) in enumerate(zip(boxes, labels, crops)):
            label = label_idx_to_name[label]
            crop_dir = Path(args.out_path) / split / label
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f'{img_path.stem}_{i}.png'
            box = box_to_square(box, max_size=max(H, W))
            x0, y0, x1, y1 = box.int().tolist()
            crop = crop[:,y0 : y1, x0 : x1]
            if crop.shape[1] < 10 or crop.shape[2] < 10:
                with open('too_small.txt', 'a') as f:
                    f.write(f'{crop_path} {crop.shape[1]} {crop.shape[2]} {label}\n')
                counter[label] += 1
                continue
            F.to_pil_image(crop).save(crop_path)
        with open('skipped.txt', 'w') as f:
            for key, value in counter.items():
                f.write(f'{key} {value}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset splitter for rg-dataset')
    parser.add_argument('--data_path', default='data',
                        help='Path to the data folder')
    parser.add_argument('--out_path', default='crops',
                        help='Path to the data folder')
    parser.add_argument('--masks',action='store_true',
                        help='Include masks in data loading')

    args = parser.parse_args()
    data_path = Path(args.data_path)
    random.seed(42)

    train_dset = build_galaxy('train', data_path, args.masks)
    val_dset = build_galaxy('val', data_path, args.masks)
    test_dset = build_galaxy('test', data_path, args.masks)

    for split in ['train', 'val', 'test']:
        crop_dataset(split)