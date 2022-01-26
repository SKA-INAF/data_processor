import math
from pathlib import Path
import random
import argparse
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset splitter for rg-dataset')
    parser.add_argument('--data_path', default='data',
                        help='Path to the data folder')
    parser.add_argument('--txt_folder', default='split_lists',
                        help='Path to the data folder')
    parser.add_argument('--out_path', default='splits',
                        help='Path to the data folder')
    parser.add_argument('--skip_meerkat',action='store_true',
                        help='Wether to skip MeerKAT data in split creation (Folders to skip have to be listed in meerkat_data.txt')

    args = parser.parse_args()
    data_path = Path(args.data_path)
    random.seed(42)

    train_perc = 0.7
    val_perc = 0.1
    test_perc = 0.2

    output_path = Path(args.out_path)
    output_path.mkdir(exist_ok=True)
    txt_folder_path = Path(args.txt_folder)
    txt_folder_path.mkdir(exist_ok=True)

    to_skip = []

    if args.skip_meerkat:
        with open('meerkat_data.txt') as mk:
            for mk_folder in mk:
                to_skip.append(mk_folder.strip())


    for class_folder in data_path.glob('*'):

        # class_folder = data/RadioGalaxies

        for sample_folder in class_folder.glob('*'):
            # sample_folder = data/RadioGalaxies/sample1

            if f'{class_folder.stem}/{sample_folder.stem}' in to_skip:
                print(f'Skipping {class_folder}/{sample_folder.stem}')
                continue

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

                with open(txt_folder_path / 'train.txt', 'a') as out:
                    for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                        t.set_description(f'Train split {class_folder.stem} - {sample_folder.stem}')
                        img_name = filename.stem # galaxy0001
                        out.write(f'{class_folder.stem}/{sample_folder.stem}/imgs/{img_name}.fits\n')
            
            with tqdm(val_split) as t:

                with open(txt_folder_path / 'val.txt', 'a') as out:
                    for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                        t.set_description(f'Val split {class_folder.stem} - {sample_folder.stem}')

                        img_name = filename.stem # galaxy0001
                        out.write(f'{class_folder.stem}/{sample_folder.stem}/imgs/{img_name}.fits\n')

            with tqdm(test_split) as t:

                with open(txt_folder_path / 'test.txt', 'a') as out:
                    for filename in t: # filename = data/RadioGalaxies/sample1/imgs/galaxy0001.fits
                        t.set_description(f'Test split {class_folder.stem} - {sample_folder.stem}')

                        img_name = filename.stem # galaxy0001
                        out.write(f'{class_folder.stem}/{sample_folder.stem}/imgs/{img_name}.fits\n')