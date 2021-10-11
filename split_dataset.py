import math
from pathlib import Path
import random
from shutil import copyfile
import argparse
import json
from tqdm import tqdm

def copy_annotations(img_name, sample_folder, masks_path, dst_json_folder):
    for json_obj_file in masks_path.glob(f'mask_{img_name}.json'):
        dst_json_path = dst_json_folder / f'{sample_folder.stem}_{json_obj_file.stem}{json_obj_file.suffix}' # splits/{phase}/sample1_mask_galaxy0001.json

        with open(json_obj_file) as j:
            data = json.load(j)
            img_name = Path(data['img'].split('/')[-1])
            renamed_img_name = f'{sample_folder.stem}_{img_name}'
            data['img'] = data['img'].replace(str(img_name), renamed_img_name)
            objs = data['objs']
            renamed_objs = []
            for obj in objs:
                if not obj['class']:
                    with open('errors.txt', 'a') as o:
                        o.write(f'Mask {obj["mask"]} has no class\n')

                renamed_mask = f'{sample_folder.stem}_{obj["mask"]}'
                obj['mask'] = renamed_mask
                renamed_objs.append(obj)

        with open(dst_json_path, 'w') as out:
            json.dump({'img': renamed_img_name, 'objs': renamed_objs}, out, indent=2)
        # copyfile(json_obj_file, dst_json_path)

def copy_masks(img_name, dst_mask_folder, masks_path):

    if not masks_path.glob(f'*mask_{img_name}*.fits'):
        with open('errors.txt', 'a') as o:
            o.write(f'Image {img_name} has no masks\n')

    for mask_obj_file in masks_path.glob(f'*mask_{img_name}*.fits'):
        dst_mask_path = dst_mask_folder / f'{sample_folder.stem}_{mask_obj_file.stem}{mask_obj_file.suffix}' # splits/{phase}/masks/sample1_mask_galaxy0001_obj1.fits
        copyfile(mask_obj_file, dst_mask_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset splitter for rg-dataset')
    parser.add_argument('--data_path', default='data',
                        help='Path to the data folder')
    parser.add_argument('--out_path', default='splits',
                        help='Path to the data folder')

    args = parser.parse_args()
    data_path = Path(args.data_path)
    random.seed(42)

    train_perc = 0.7
    val_perc = 0.1
    test_perc = 0.2

    output_path = Path(args.out_path)

    dst_train_folder = output_path / 'train'
    dst_img_train_folder = output_path / 'train' / 'imgs'
    dst_mask_train_folder = output_path / 'train' / 'masks'
    dst_json_train_folder = output_path / 'train' / 'annotations'

    dst_val_folder = output_path / 'val'
    dst_img_val_folder = output_path / 'val' / 'imgs'
    dst_mask_val_folder = output_path / 'val' / 'masks'
    dst_json_val_folder = output_path / 'val' / 'annotations'

    dst_test_folder = output_path / 'test'
    dst_img_test_folder = output_path / 'test' / 'imgs'
    dst_mask_test_folder = output_path / 'test' / 'masks'
    dst_json_test_folder = output_path / 'test' / 'annotations'

    output_path.mkdir(exist_ok=True)

    dst_train_folder.mkdir(exist_ok=True)
    dst_img_train_folder.mkdir(exist_ok=True)
    dst_mask_train_folder.mkdir(exist_ok=True)
    dst_json_train_folder.mkdir(exist_ok=True)

    dst_val_folder.mkdir(exist_ok=True)
    dst_img_val_folder.mkdir(exist_ok=True)
    dst_mask_val_folder.mkdir(exist_ok=True)
    dst_json_val_folder.mkdir(exist_ok=True)

    dst_test_folder.mkdir(exist_ok=True)
    dst_img_test_folder.mkdir(exist_ok=True)
    dst_mask_test_folder.mkdir(exist_ok=True)
    dst_json_test_folder.mkdir(exist_ok=True)


    for class_folder in data_path.glob('*'):

        # class_folder = data/RadioGalaxies

        for sample_folder in class_folder.glob('*'):

            # sample_folder = data/RadioGalaxies/sample1

            imgs_path = sample_folder / 'imgs' # data/RadioGalaxies/sample1/imgs
            masks_path = sample_folder / 'masks' # data/RadioGalaxies/sample1/masks

            img_file_list = sorted(imgs_path.glob('*.fits'))
            mask_file_list = sorted(masks_path.glob('*.fits'))

            if not mask_file_list:
                continue
            random.shuffle(img_file_list)

            train_samples = math.floor(train_perc * len(img_file_list))
            train_split = img_file_list[ : train_samples]

            val_samples = math.floor(val_perc * len(img_file_list))
            val_split = img_file_list[train_samples : train_samples + val_samples]

            test_split = img_file_list[train_samples + val_samples : ]

            with tqdm(train_split) as t:

                for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                    t.set_description(f'Train split {class_folder.stem} - {sample_folder.stem}')
                    img_name = filename.stem # galaxy0001
                    dst_img_path = dst_img_train_folder / f'{sample_folder.stem}_{filename.stem}{filename.suffix}' # splits/train/imgs/sample1_galaxy0001.fits

                    copy_annotations(img_name, sample_folder, masks_path, dst_json_train_folder)
                    copy_masks(img_name, dst_mask_train_folder, masks_path)
                    copyfile(filename, dst_img_path)
            
            with tqdm(val_split) as t:

                for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                    t.set_description(f'Val split {class_folder.stem} - {sample_folder.stem}')

                    img_name = filename.stem # galaxy0001
                    dst_img_path = dst_img_val_folder / f'{sample_folder.stem}_{filename.stem}{filename.suffix}' # splits/val/imgs/sample1_galaxy0001.fits

                    copy_annotations(img_name, sample_folder, masks_path, dst_json_val_folder)
                    copy_masks(img_name, dst_mask_val_folder, masks_path)
                    copyfile(filename, dst_img_path)

            with tqdm(test_split) as t:

                for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                    t.set_description(f'Test split {class_folder.stem} - {sample_folder.stem}')

                    img_name = filename.stem # galaxy0001
                    dst_img_path = dst_img_test_folder / f'{sample_folder.stem}_{filename.stem}{filename.suffix}' # splits/test/imgs/sample1_galaxy0001.fits

                    copy_annotations(img_name, sample_folder, masks_path, dst_json_test_folder)
                    copy_masks(img_name, dst_mask_test_folder, masks_path)
                    copyfile(filename, dst_img_path)





