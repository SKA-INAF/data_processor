from genericpath import exists
import random, json, os, math
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from pathlib import Path

from parsers.base import DefaultParser

class COCOParser(DefaultParser):

    def __init__(self, contrast, dst_dir, split):
        super(COCOParser, self).__init__(dst_dir, split)
        self.contrast = contrast

    def make_img_dir(self, entries):
        '''Copies images into train or val folder'''
        split_path = self.dst_dir / Path(self.split)
        split_path.mkdir(exist_ok=True)
        dst_folder = split_path #/  Path('imgs')
        for line in tqdm(entries):
            img_name = line['img'].name # take image name
            img_name = img_name.replace('.fits', '.png')
            dst_folder.mkdir(exist_ok=True)
            dst_path = dst_folder / img_name
            line['filename'] = img_name
            self.fits_to_png(line['img'], dst_path, contrast=self.contrast)

    def make_annotations(self, samples, incremental_id):
        '''Creates the JSON COCO annotations to be stored'''

        coco_samples = { 'images':[], 'annotations':[], 'categories': [
                                                            {"id":1, "name": 'sidelobe'},
                                                            {"id":2, "name":'source'},
                                                            {"id":3, "name":'galaxy'},
                                                                ] 
                            }

        for line in tqdm(samples):

            h, w = fits.getdata(line['img']).shape
            image = {'id': incremental_id['img'], 'width': w, 'height': h, 'file_name': line['img'].name.replace('.fits', '.png')} # file_name = sample1_galaxy0001.png
            coco_samples['images'].append(image)

            for obj in line['objs']:
                if obj['class'] == '':
                    # probably for misannotation, the class is missing in some samples, which will be skipped 
                    self.log_error(f'Object {obj["mask"]} has no class')
                    continue
                # replaces the last two steps of the path with the steps to reach the mask file
                mask_path = line['img'].parent.parent / Path('masks') / obj['mask']
                x_points, y_points = self.get_mask_coords(mask_path)

                if not (x_points and y_points):
                    self.log_error(f'Mask {mask_path} is empty')
                    continue

                poly = [(x, y) for x, y in zip(x_points, y_points)]
                # Flatten the array
                poly = [p for x in poly for p in x]

                if len(poly) <= 4:
                    # Eliminates annotations with segmentation masks with only 2 coordinates,
                    # which bugs the coco API
                    id = image['id']
                    filename = image['file_name']
                    msg = f'Invalid mask for file: {filename}\tlen: {len(poly)} (should be > 4)\t objs: {len(line["objs"])}'
                    self.log_error(msg)
                    continue

                x0, y0, x1, y1 = np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points) 
                w, h = x1 - x0, y1 - y0 
                area = w * h
                
                annotation = {
                    'id': incremental_id['obj'], 
                    'category_id': self.CLASSES[obj['class']],
                    'image_id': incremental_id['img'], 
                    'segmentation': [poly],
                    'area': area,
                    "bbox": [x0, y0, w, h],
                    'iscrowd': 0
                }

                coco_samples['annotations'].append(annotation)

                incremental_id.update({'obj': 1})

            incremental_id.update({'img': 1})

        return coco_samples


    def dump_annotations(self, coco_samples):
        annot_dst_dir = self.dst_dir / Path('annotations')
        annot_dst_dir.mkdir(exist_ok=True)
        annot_dst_path = annot_dst_dir / Path(f'{self.split}.json')
        with open(annot_dst_path, 'w') as out:
            print(f'Dumping data in file {self.split}.json')
            json.dump(coco_samples, out, indent=2, cls=self.json_enc)