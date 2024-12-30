import numpy as np
import os
import argparse
import h5py

def convert_vtab_hdf5(source_path, target_path):
    h5_file = h5py.File(target_path, 'w')
    for p in os.listdir(source_path):
        if p.startswith('.'): continue
        sub_path = os.path.join(source_path, p)
        img_name_list = [fn for fn in os.listdir(sub_path) if not fn.startswith('.')]
        h5_group = h5_file.create_group(p)
        for img_name in sorted(img_name_list):
            with open(os.path.join(sub_path, img_name), 'rb') as f:
                img_obj = np.asarray(f.read())
            img_key = img_name
            h5_group.create_dataset(img_key, data=img_obj)
    h5_file.close()

def convert_fewshot_hdf5(source_path, target_path, annotation_path):
    filter_image_list = []
    for list_file in os.listdir(annotation_path):
        if list_file.startswith('.'): continue
        with open(os.path.join(annotation_path, list_file)) as f:
            filter_image_list += [l.strip().rsplit(' ', 1)[0] for l in f.readlines()]
    filter_image_list = set(filter_image_list)
    h5_file = h5py.File(target_path, 'w')
    for img_name in sorted(filter_image_list):
        with open(os.path.join(source_path, img_name), 'rb') as f:
            img_obj = np.asarray(f.read())
        img_key = img_name
        h5_file.create_dataset(img_key, data=img_obj)
    h5_file.close()

############ usages example
### For vtab-1k:
# python convert_to_hdf5.py --src data/vtab-1k/cifar/images --dst data/vtab-1k/cifar/images.hdf5

### For few-shot:
# python convert_to_hdf5.py \
# --src data/few-shot/oxford-flowers102/images \
# --dst data/few-shot/oxford-flowers102/images.hdf5 \
# --task fs --annotation_path data/few-shot/oxford-flowers102/annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert jpeg images to jpeg bytes stream list and store into hdf5 file')
    parser.add_argument('--src', type=str, default=None, required=True, help='directory path of images')
    parser.add_argument('--dst', type=str, default=None, required=True, help='hdf5 file path')
    parser.add_argument('--annotation_path', type=str, default=None, required=False, help='Annotation list for dataset')
    parser.add_argument('--task', type=str, default='vtab', choices=['vtab', 'fs'],
                        help='choice from vtab: vtab-1k, fs: few-shot')
    args = parser.parse_args()

    if   args.task == 'vtab': convert_vtab_hdf5(args.src, args.dst)
    elif args.task == 'fs': convert_fewshot_hdf5(args.src, args.dst, args.annotation_path)