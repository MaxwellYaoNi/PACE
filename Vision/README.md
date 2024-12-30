# Official PyTorch Implementation of PACE for VTAB-1K and Few-Shot Learning

## Overview
Key components include:

-`pace/pace_ops.py`: Code for applying noise to adapters and implementing consistency regularization.

-`pace/residual_adapters.py`: Code for Residual Adapters.

-`train.py (Line 137)`: Example of how to inject PACE into the training process.

---

## 1. Download ViT-B
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and place it in the root folder.

## 2. Download datasets
Follow [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation) to download the dataset. Then move the dataset folders to `data/`.

## 3. [Optional] Convert data to HDF5.
If your computing resources are limited by the number of files, you can group them into a .hdf5 file.
```
python3 convert_to_hdf5.py --src data/vtab-1k/cifar/images --dst data/vtab-1k/cifar/images.hdf5
```
## 4. Training Scripts

### 4.1 VTAB-1k (CIFAR)
#### Baseline (LoRAmul_VPTadd):
Remove `--hdf5` if you haven't convert files to HDF5.
```
python3 train.py \
--dataset cifar --lr 1e-03 --wd 1e-4 --rank 10 --epoch 300 --hdf5
```
#### LoRAmul_VPTadd with PACE:
```
python3 train.py \
--dataset cifar --lr 1e-03 --wd 1e-4 --rank 10 --epoch 300 --hdf5 \
--pace_type pace --lbd 1 --sigma 1.2 
```

### 4.2 Few-Shot Learning (Oxford-Flowers102)
#### Basline (LoRAmul_VPTadd)
```
python3 train.py \
--task fs --dataset oxford-flowers102 --lr 5e-03 --wd 1e-4 --rank 18 --epoch 100 --hdf5
```

## 5. Notes and Tips
### Argument Details
`--model`: Use `--model Swin-B` to replace the pre-trained model with Swin-B

`--adapter`: Change adapters with options like `LoRAadd` or `VPTadd` (e.g., `--adapter LoRAadd`).

`--pace_type`: Options are`pace`,`lazy`, and`fast`. `pace`offers the best performance. `lazy`and `fast` are faster but need adjustments: `lazy` requires a larger `--lbd`, while `fast` needs a larger `--sigma` and smaller `--lbd`.
### Hyperparameter Tuning
Adjust `--sigma` and `--lbd` for different datasets. For smaller datasets, larger values generally yield better results.
