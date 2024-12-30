import h5py, json, io, os
from PIL import Image
import numpy as np
import torch.utils.data as data

def json_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        json_data = json.load(rf)
    for imkey in sorted(json_data.keys()):
        imlabel = json_data[imkey]
        imlist.append((imkey, int(imlabel)))

    class_ids = sorted(list(set([l for _, l in imlist])))
    class_mapping = {v: i for i, v in enumerate(class_ids)}
    imlist = [(imkey, class_mapping[imlabel]) for imkey, imlabel in imlist]
    return imlist


def h5_loader(data_source, imkey):
    return Image.open(io.BytesIO(np.array(data_source[imkey]))).convert("RGB")

def file_loader(data_source, imkey):
    return Image.open(os.path.join(data_source, imkey)).convert("RGB")

class ImageHdf5Data(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, flist_reader=json_flist_reader,
                 is_hdf5=True, return_index=False):
        self.data_source = h5py.File(root, 'r') if is_hdf5 else root
        self.loader = h5_loader if is_hdf5 else file_loader
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index

    def __getitem__(self, index):
        imkey, target = self.imlist[index]
        img = self.loader(self.data_source, imkey)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return img, target, index
        return img, target

    def __len__(self):
        return len(self.imlist)