from pathlib import Path
import random
import json
from collections import Counter

def count_objects(folder, counter, fto, phase='Train'):

    n_objs_per_img = []
    
    for mask_json in sorted(folder.glob('*.json')):
        with open(mask_json) as m:
            objects = json.load(m)['objs']
            for obj in objects:
                counter.update({f"{phase.lower()}_{obj['class']}": 1})

        n_objs = len(objects)
        fto.write(f'{mask_json.stem}: {n_objs}\n')

        n_objs_per_img.append(n_objs)

    return n_objs_per_img

if __name__ == '__main__':
    data_path = 'data'
    data_path = Path(data_path)
    random.seed(42)

    c = Counter()

    train_perc = 0.7
    val_perc = 0.1
    test_perc = 0.2

    output_path = 'splits'
    output_path = Path(output_path)

    dst_train_folder = output_path / 'train'
    dst_img_train_folder = output_path / 'train' / 'imgs'
    dst_mask_train_folder = output_path / 'train' / 'annotations'

    dst_val_folder = output_path / 'val'
    dst_img_val_folder = output_path / 'val' / 'imgs'
    dst_mask_val_folder = output_path / 'val' / 'annotations'

    dst_test_folder = output_path / 'test'
    dst_img_test_folder = output_path / 'test' / 'imgs'
    dst_mask_test_folder = output_path / 'test' / 'annotations'

    fto = Path('file_to_objs.txt')

    if fto.exists():
        fto.unlink()

    n_objs = Path('n_objs_per_img.txt')

    if n_objs.exists():
        n_objs.unlink()

    with open(fto, 'a') as out:
        out.write('Train\n')
        n_objs_per_img = count_objects(dst_mask_train_folder, c, out, 'Train')
    with open(n_objs, 'a') as out:
        out.write('Train\n')
        out.write(f'Max objects per image: {max(n_objs_per_img)} \n')
        out.write(f'Min objects per image: {min(n_objs_per_img)} \n')
        out.write(f'Avg objects per image: {sum(n_objs_per_img) / len(n_objs_per_img)} \n')
    
    with open(fto, 'a') as out:
        out.write('Val\n')
        n_objs_per_img = count_objects(dst_mask_val_folder, c, out, 'Val')
    with open(n_objs, 'a') as out:
        out.write('Val\n')
        out.write(f'Max objects per image: {max(n_objs_per_img)} \n')
        out.write(f'Min objects per image: {min(n_objs_per_img)} \n')
        out.write(f'Avg objects per image: {sum(n_objs_per_img) / len(n_objs_per_img)} \n')

    with open(fto, 'a') as out:
        out.write('Test\n')
        n_objs_per_img = count_objects(dst_mask_test_folder, c, out, 'Test')
    with open(n_objs, 'a') as out:
        out.write('Test\n')
        out.write(f'Max objects per image: {max(n_objs_per_img)} \n')
        out.write(f'Min objects per image: {min(n_objs_per_img)} \n')
        out.write(f'Avg objects per image: {sum(n_objs_per_img) / len(n_objs_per_img)} \n')

    with open('obj_count.json', 'w') as out:
        json.dump(c, out)


