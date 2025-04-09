import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from config import exp_root
from models.model import DINOHead_k
from models.model import ContrastiveLearningViewGenerator, get_params_groups
from my_utils.ood_utils import get_ood_scores_in, get_ood_scores, get_measures, print_measures, write_measures, print_measures_with_std, write_measures_with_std


def get_and_print_results(ood_loader, model, in_score, args):
    aurocs, auprs_in, auprs_out, fprs_in, fprs_out = [], [], [], [], []

    for _ in range(args.num_to_avg):
        out_score = get_ood_scores(ood_loader, model, OOD_NUM_EXAMPLES, args)
        measures_in = get_measures(-in_score, -out_score)
        measures_out = get_measures(out_score, in_score)  # OE's defines out samples as positive

        auroc = measures_in[0]; aupr_in = measures_in[1]; aupr_out = measures_out[1]; fpr_in = measures_in[2]; fpr_out = measures_out[2]
        aurocs.append(auroc); auprs_in.append(aupr_in); auprs_out.append(aupr_out); fprs_in.append(fpr_in); fprs_out.append(fpr_out)

    if args.num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs_in, auprs_out, fprs_in, fprs_out)
        write_measures_with_std(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, file_path=args.ood_log_path)
    else:
        print_measures(np.mean(aurocs), np.mean(auprs_in), np.mean(auprs_out), np.mean(fprs_in), np.mean(fprs_out))
        write_measures(np.mean(aurocs), np.mean(auprs_in), np.mean(auprs_out), np.mean(fprs_in), np.mean(fprs_out), file_path=args.ood_log_path)

    return (auroc, aupr_in, aupr_out, fpr_in, fpr_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--ckpts_date', type=str, default=None)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    #parser.add_argument('--init_prototypes', action='store_true', default=False)

    #parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--ood_log_path', type=str, default='OOD_results')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--score', type=str, default='msp', help='OOD detection score function: [msp, mls, energy, xent]')
    parser.add_argument('--temp_logits', default=0.1, type=float, help='cosine similarity of prototypes to classification logits temperature')
    parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
    parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    #init_experiment(args, runner_name=['ProtoGCD'])
    #args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    args.ood_log_path = os.path.join(args.ood_log_path, args.dataset_name)
    if not os.path.exists(args.ood_log_path):
        os.makedirs(args.ood_log_path)
    args.ood_log_path = os.path.join(args.ood_log_path, args.ckpts_date + '-' + args.score + '-T' + str(args.temp_logits) + '.txt')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # if args.warmup_model_dir is not None:
    #     args.logger.info(f'Loading weights from {args.warmup_model_dir}')
    #     backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets_ = get_datasets(args.dataset_name,
                                                                                          train_transform,
                                                                                          test_transform,
                                                                                          args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    # train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
    #                           sampler=sampler, drop_last=True, pin_memory=True)
    # test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    OOD_NUM_EXAMPLES = len(test_dataset) // 5   # NOTE! NOT test_loader_labelled!
    print(OOD_NUM_EXAMPLES)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead_k(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers,
                           init_prototypes=None, num_labeled_classes=args.num_labeled_classes)
    model = nn.Sequential(backbone, projector).to(device)

    ckpts_base_path = '/lustre/home/sjma/GCD-project/protoGCD-v7/dev_outputs_fix/'
    ckpts_path = os.path.join(ckpts_base_path, args.dataset_name, args.ckpts_date, 'checkpoints', 'model_best.pt')
    ckpts = torch.load(ckpts_path)
    ckpts = ckpts['model']
    print('loading ckpts from %s...' % ckpts_path)
    model.load_state_dict(ckpts)
    print('successfully load ckpts')
    model.eval()


    # ----------------------
    # TEST OOD
    # ----------------------
    print('Using %s as typical data' % args.dataset_name)
    with open(args.ood_log_path, 'w+') as f_log:
        f_log.write('Using %s as typical data' % args.dataset_name)
        f_log.write('\n')

    print(test_transform)

    # ID score
    #test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    in_score, right_score, wrong_score = get_ood_scores_in(test_loader_labelled, model, args)


    # Textures
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/dtd/images", transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nTexture Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nTexture Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)


    # SVHN
    ood_data = datasets.SVHN('/data4/sjma/dataset/SVHN/', split='test', download=False, transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nSVHN Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nSVHN Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)

    # Places
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/places365", transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nPlaces Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nPlaces Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)


    # TinyImageNet-R
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/Imagenet_resize", transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nTinyImageNet-resize Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nTinyImageNet-resize Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)


    # LSUN-R
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/LSUN_resize", transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nLSUN-resize Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nLSUN-resize Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)


    # iSUN
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/iSUN", transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\niSUN Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\niSUN Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)


    # CIFAR data
    if args.dataset_name == 'cifar10':
        ood_data = datasets.CIFAR100('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)
    else:
        ood_data = datasets.CIFAR10('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('\n\nCIFAR-100 Detection') if args.dataset_name == 'cifar10' else print('\n\nCIFAR-10 Detection')
    with open(args.ood_log_path, 'a+') as f_log:
        f_log.write('\n\nCIFAR-100 Detection') if args.dataset_name == 'cifar10' else f_log.write('\n\nCIFAR-10 Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader, model, in_score, args)
