import torchvision.transforms as transforms
import torch, os
from .h5dataset import ImageHdf5Data

_vtab_class_num_dict = {'cifar':                100,
                        'caltech101':           102,
                        'dtd':                  47,
                        'oxford_flowers102':    102,
                        'oxford_iiit_pet':      37,
                        'svhn':                 10,
                        'sun397':               397,
                        'patch_camelyon':       2,
                        'eurosat':              10,
                        'resisc45':             45,
                        'diabetic_retinopathy': 5,
                        'clevr_count':          8,
                        'clevr_dist':           6,
                        'dmlab':                6,
                        'kitti':                4,
                        'dsprites_loc':         16,
                        'dsprites_ori':         16,
                        'smallnorb_azi':        18,
                        'smallnorb_ele':        9}

def get_vtab_data(name, evaluate=False, resize=224, batch_size=64, num_workers=8, is_hdf5=True):
    if name in _vtab_class_num_dict:
        root = os.path.join('data', 'vtab-1k', name)
        transform = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_root = os.path.join(root, 'images.hdf5' if is_hdf5 else 'images')

        def flist_reader(flist):
            imlist = []
            with open(flist, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel = line.strip().rsplit(' ', 1)
                    impath = impath.split('/', 1)[-1]
                    imlist.append((impath, int(imlabel)))
            return imlist

        if evaluate:
            train_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "train800val200.txt"), transform=transform,
                              flist_reader=flist_reader, return_index=True, is_hdf5=is_hdf5),
                batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "test.txt"), transform=transform,
                              flist_reader=flist_reader, is_hdf5=is_hdf5),
                batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "train800.txt"), transform=transform,
                              flist_reader=flist_reader, return_index=True, is_hdf5=is_hdf5),
                batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "val200.txt"), transform=transform,
                              flist_reader=flist_reader, is_hdf5=is_hdf5),
                batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader
    else:
        raise NotImplementedError(f'VTAB-1K does not have dataset: {name}')


def get_vtab_classes_num(dataset_name):
    return _vtab_class_num_dict[dataset_name]