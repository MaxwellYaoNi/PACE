import os
from .h5dataset import ImageHdf5Data
import torch
import torchvision.transforms as transforms

def get_few_shot_data(dataset_name, batch_size=64, num_workers=8, shot=1, seed=0, is_hdf5=True):
    root = os.path.join('data', 'few-shot', dataset_name)
    image_root  = os.path.join(root, 'images.hdf5' if is_hdf5 else 'images')
    flist_train = os.path.join(root, 'annotations', f'train_meta.list.num_shot_{shot}.seed_{seed}')
    flist_val   = os.path.join(root, 'annotations', 'val_meta.list')
    flist_test  = os.path.join(root, 'annotations', 'test_meta.list')

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def flist_reader(flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                impath, imlabel = line.strip().rsplit(' ', 1)
                imlist.append((impath, int(imlabel)))
        class_ids = sorted(list(set([l for _, l in imlist])))
        assert max(class_ids) == len(class_ids) - 1
        assert len(class_ids) == get_few_shot_classes_num(dataset_name)
        class_mapping = {v: i for i, v in enumerate(class_ids)}
        imlist = [(imkey, class_mapping[imlabel]) for imkey, imlabel in imlist]
        return imlist

    train_loader = torch.utils.data.DataLoader(
        ImageHdf5Data(root=image_root, flist=flist_train, transform=train_transform, flist_reader=flist_reader,
                      is_hdf5=is_hdf5, return_index=True),
        batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageHdf5Data(root=image_root, flist=flist_val, transform=val_transform, flist_reader=flist_reader, is_hdf5=is_hdf5),
        batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageHdf5Data(root=image_root, flist=flist_test, transform=val_transform, flist_reader=flist_reader, is_hdf5=is_hdf5),
        batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_few_shot_classes_num(dataset_name):
    return {'fgvc-aircraft': 100,
            'food101': 101,
            'oxford-flowers102': 102,
            'oxford-pets': 37,
            'stanford-cars': 196
            }[dataset_name]