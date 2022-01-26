import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from parsers.base import DefaultParser

class YOLOParser(DefaultParser):

    def __init__(self, contrast, dst_dir, split):
        super(YOLOParser, self).__init__(dst_dir, split)
        self.contrast = contrast

    def make_img_dir(self, entries):
        '''Copies images into train or val folder'''
        image_dir = self.dst_dir / Path('images')
        image_dir.mkdir(exist_ok=True)
        image_split_dir = image_dir / Path(self.split)
        image_split_dir.mkdir(exist_ok=True)

        with open(f'{self.split}.txt', 'w') as txt:
            for line in tqdm(entries):
                img_name = line['img'].stem
                dst_path = (image_split_dir / Path(img_name)).with_suffix('.png')
                line['filename'] = img_name
                txt.write(str(dst_path) + '\n')
                self.fits_to_png(line['img'], dst_path, contrast=self.contrast)

    def make_annotations(self, samples, incremental_id):
        '''Creates the text file annotations to be stored'''

        dst_dir = self.dst_dir / Path('labels')
        dst_dir.mkdir(exist_ok=True)
        dst_dir = dst_dir / Path(self.split)
        dst_dir.mkdir(exist_ok=True)

        for line in tqdm(samples):

            dst_path = (dst_dir / Path(line['filename']).with_suffix('.txt'))

            with open(dst_path, 'w') as obj_file:

                for obj in line['objs']:
                    if obj['class'] == '':
                        # probably for misannotation, the class is missing in some samples, which will be skipped 
                        continue
                    # replaces the last two steps of the path with the steps to reach the mask file
                    mask_path = os.path.join(os.sep.join(str(line['img']).split(os.sep)[:-2]), 'masks', obj['mask'])
                    x_points, y_points = self.get_mask_coords(mask_path)

                    if not (x_points and y_points):
                        self.log_error(f'Mask {mask_path} is empty')
                        continue

                    w, h = self.get_img_size(mask_path)

                    x_center = (np.max(x_points) + np.min(x_points)) / 2
                    y_center = (np.max(y_points) + np.min(y_points)) / 2
                    box_width = np.max(x_points) - np.min(x_points)
                    box_height = np.max(y_points) - np.min(y_points)

                    # Normalize coordinates
                    x_center = x_center / w
                    y_center = y_center / h
                    box_width = box_width / w 
                    box_height = box_height / h


                    if x_center < 0 or y_center < 0 or \
                        box_width < 0 or box_height < 0:
                        self.log_error(f'Box format for {mask_path} is invalid')
                        continue

                    obj_file.write(f'{self.CLASSES[obj["class"]]} {x_center} {y_center} {box_width} {box_height}\n')

                    incremental_id.update({'obj': 1})

                incremental_id.update({'img': 1})

    def make_data_file(self):
        data_file = self.dst_dir / Path('radiogalaxy.yaml')
        with open(data_file, 'w') as out:
            out.write('# Number of classes')
            out.write(f'\nnc: {len(self.CLASSES)}\n')
            out.write('\n# Train and val directories')
            out.write(f'\ntrain: data/images/train/')
            out.write(f'\nval: data/images/val/')
            out.write(f'\nnames: [ ')
            for name in self.CLASSES:
                out.write(f'\'{name}\', ')
            out.write(f']')