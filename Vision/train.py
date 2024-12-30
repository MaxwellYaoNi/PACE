import os.path
from torch.optim import AdamW
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from typing import Dict
import functools
import numpy as np

import utils
import pace


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc_list = []
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        acc_list.append(logits.argmax(dim=1) == y)
    return (torch.cat(acc_list) * 1.).mean().item()

def filter_loss(loss_dict: Dict):
    new_dict = {}
    for k, v in loss_dict.items():
        if 'loss' in k:
            new_dict[k] = v.item()
    return new_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    group = parser.add_argument_group('tasks arguments')
    group.add_argument('--task',            type=str,   default='vtab', choices=['vtab', 'fs'], help='task name, vtab or fs(few-shot learning)')
    group.add_argument('--dataset',         type=str,   default='caltech101', help='dataset name')
    group.add_argument('--fs_shot',         type=int,   default=1, help='number of shot')
    group.add_argument('--fs_seed',         type=int,   default=0)
    group.add_argument('--hdf5',            action='store_true',  default=False)

    group = parser.add_argument_group('model arguments and training arguments')
    group.add_argument('--model',           type=str,   default='ViT-B', choices=['ViT-B', 'Swin-B'])
    group.add_argument('--bs',              type=int,   default=64)
    group.add_argument('--lr',              type=float, default=1e-3)
    group.add_argument('--wd',              type=float, default=1e-4)
    group.add_argument('--num_workers',     type=int,   default=4)
    group.add_argument('--epoch',           type=int,   default=300)
    group.add_argument('--test_every',      type=int,   default=10, help='testing after specific epochs')

    group = parser.add_argument_group('working settings')
    group.add_argument('--seed',            type=int,   default=42)
    group.add_argument('--out_dir',         type=str,   default='outs')

    group = parser.add_argument_group('adapter settings')
    group.add_argument('--adapter',         type=str,   default='LoRAmul_VPTadd', choices=['LoRAmul_VPTadd', 'LoRAadd', 'VPTadd'])
    group.add_argument('--rank',            type=int,   default=10)

    group = parser.add_argument_group('PACE arguments')
    group.add_argument('--pace_type',      type=str,   default=None, choices=[None, 'pace', 'lazy', 'fast'])
    group.add_argument('--lbd',            type=float, default=0., help='regularization strength of PACE')
    group.add_argument('--sigma',          type=float, default=0., help='sigma of the Gaussian noise')
    group.add_argument('--lazy_interval',  type=int,   default=1,  help="It only works when pace_type is set to 'lazy'")


    args = parser.parse_args()

    # set the working directory
    name_configs = []
    if args.pace_type is not None:
        name_configs += [args.pace_type,
                         f'Lbd{args.lbd:g}'         if args.lbd else None,
                         f'S{args.sigma:g}'         if args.sigma else None,
                         f'Lz{args.pace_every}'     if args.pace_type == 'lazy' and args.lazy_interval != 1 else None]

    name_configs += [    args.adapter,
                         f'R{args.rank:d}',
                         f'lr{args.lr:g}'                       if args.lr != 1e-3 else None,
                         f'wd{args.wd:.0e}'                     if args.wd != 1e-4 else None,
                         f'St{args.fs_shot}Sd{args.fs_seed}'    if args.task == 'fs' else None,
                         args.dataset]

    base_sub_path           = '_'.join([nc for nc in name_configs if nc is not None])
    args.save_path          = os.path.join(args.out_dir, base_sub_path)
    utils.ensure_dirs(args.save_path)
    args.save_path_recent   = os.path.join(args.save_path, 'weight.pt')

    print(args)
    utils.set_seed(args.seed)
    args.best_acc = 0

    # set the loggers
    test_logger = utils.MetricsLogger(args.save_path + '_acc', True, 1)
    train_logger = utils.TrainLogger(args.save_path, True, buffer_size=100)

    # load dataset
    if args.task == 'vtab':
        train_dl, val_dl = utils.get_vtab_data(args.dataset, False, batch_size=args.bs,
                                               num_workers=args.num_workers, is_hdf5=args.hdf5)
        test_dl = None
        class_dim = utils.get_vtab_classes_num(args.dataset)
    elif args.task == 'fs':
        train_dl, val_dl, test_dl = utils.get_few_shot_data(args.dataset, batch_size=args.bs, num_workers=args.num_workers,
                                                            shot=args.fs_shot, seed=args.fs_seed, is_hdf5=args.hdf5)
        class_dim = utils.get_few_shot_classes_num(args.dataset)
        args.test_acc = 0
    else:
        raise NotImplementedError


    # load the pretrained model
    if args.model == 'ViT-B':
        model = create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    elif args.model == 'Swin-B':
        import timm
        timm.models._hub.hf_hub_download = functools.partial(timm.models._hub.hf_hub_download, cache_dir='cache')
        model = create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=True, drop_path_rate=0.1)
    else:
        model = None

    model.reset_classifier(class_dim)
    to_cuda = torch.cuda.is_available()

    # decide how to compute the loss
    def compute_loss_standard(model, x, y, criterion, **kwargs):
        logits = model(x)
        cls_loss = criterion(logits, y)
        return {'cls_loss': cls_loss, 'total_loss': cls_loss, 'logits': logits}

    compute_loss = compute_loss_standard
    cross_entropy = torch.nn.CrossEntropyLoss()

    # inject residual adapter
    pace.inject_residual_adapter(model, adapter=args.adapter, rank=args.rank)

    ############# start of injecting PACE #################################
    if args.pace_type is not None:
        # PACE performs two forward passes on the same sample.
        # Ensure DropPath use the same mask for both passes to maintain consistency
        pace.ensure_sharable_drop_path(model)

        ######### STEP 1: inject noise to the adapters
        # get list of [parent_layer, adapter_name, block_id] for each ResidualAdapter
        adapters_and_block_ids = pace.get_adapters_and_block_ids(model)

        # sigma value decreases linearly as block_id increases.
        num_blocks = max(block_id for _, _, block_id in adapters_and_block_ids) + 1
        block_start = 1 if args.model == 'ViT-B' else 0 # ViT-B works better without adding noise to the first block
        sigmas = np.concatenate([np.zeros(block_start), np.linspace(0, args.sigma, num_blocks - block_start + 1)[-1:0:-1]])

        # Inject noise adapter into each residual adapter
        for parent_layer, adapter_name, block_id in adapters_and_block_ids:
            residual_adapter = getattr(parent_layer, adapter_name)
            noise_adapter = pace.MultiplicativeNoiseAdapter(residual_adapter, sigma=sigmas[block_id])
            setattr(parent_layer, adapter_name, noise_adapter)

        ######### STEP 2: apply consistency regularization over two perturbations
        if args.pace_type == 'pace':  # PACE: applying consistency regularization at every iteration
            compute_loss = functools.partial(pace.compute_loss_pace, lbd_pace=args.lbd)
        elif args.pace_type == 'lazy':  # PACE_lazy_half: apply consistency regularization at every 'lazy_interval' iterations and halve the batch size
            compute_loss = functools.partial(pace.compute_loss_pace_lazy_half, lbd_pace=args.lbd, lazy_interval=args.lazy_interval)
        elif args.pace_type == 'fast':  # PACE_fast
            history_out = torch.zeros(len(train_dl.dataset), class_dim)
            if to_cuda: history_out = history_out.cuda()
            compute_loss = functools.partial(pace.compute_loss_pace_fast, lbd_pace=args.lbd, history_logits=history_out)
    ############# end of injecting PACE  #################################

    # set optimizer and scheduler
    trainable = []
    trainable_names = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['delta', 'head']]):
            trainable.append(p)
            p.requires_grad = True
            trainable_names[n] = p
        else:
            p.requires_grad = False
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=args.epoch, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)

    # train the model
    if to_cuda: model = model.cuda()
    itr = 0
    for ep in tqdm(range(args.epoch)):
        model.train()
        for i, (x, y, index) in enumerate(train_dl):
            # forward and backward
            if to_cuda: x, y, index = x.cuda(), y.cuda(), index.cuda()
            out_dict = compute_loss(model, x, y, cross_entropy, itr=itr, index=index)
            loss = out_dict['total_loss']
            opt.zero_grad()
            loss.backward()
            opt.step()

            # log training results
            results_dict = filter_loss(out_dict)
            with torch.no_grad():
                out = out_dict['logits']
                results_dict['train_acc'] = torch.mean((out.argmax(dim=1) == y[:out.shape[0]]) * 1.).item()
            train_logger.log(itr, **results_dict)

            itr += 1
            torch.cuda.empty_cache()

        scheduler.step(ep)

        # on validation
        if (ep+1) % args.test_every == 0 or (ep+1) == args.epoch:
            acc = round(test(model, val_dl), 4)
            utils.save_weights(args.save_path_recent, model, trainable_names)
            val_results = {'val_acc': acc}
            if test_dl is not None:
                test_acc = round(test(model, test_dl), 4)
                val_results['test_acc'] = test_acc
            test_logger.log(E=ep+1, **val_results)
