import os
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from my_utils.general_utils import AverageMeter, init_experiment
from my_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root

from models.model import DINOHead
from models.model import ContrastiveLearningViewGenerator, get_params_groups
from models.loss import info_nce_logits, SupConLoss, DistillLoss_ratio, prototype_separation_loss, entropy_regularization_loss




def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )


    distill_criterion = DistillLoss_ratio(num_classes=args.num_labeled_classes + args.num_unlabeled_classes,
                                          wait_ratio_epochs=args.wait_ratio_epochs,
                                          ramp_ratio_teacher_epochs=args.ramp_ratio_teacher_epochs,
                                          nepochs=args.epochs,
                                          ncrops=args.n_views,
                                          init_ratio=args.init_ratio,
                                          final_ratio=args.final_ratio,
                                          temp_logits=args.temp_logits,
                                          temp_teacher_logits=args.temp_teacher_logits,
                                          device=device)

    # inductive
    #best_test_acc_ubl = 0
    best_test_acc_lab = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0
    best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)


            student_proj, student_out, prototypes = student(images)
            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / args.temp_logits).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup
            cluster_loss = 0
            distill_loss = distill_criterion(student_out, teacher_out, epoch)   # NOTE!!! all data
            cluster_loss += distill_loss

            entropy_reg_loss = entropy_regularization_loss(student_out, args.temp_logits)
            cluster_loss += args.weight_entropy_reg * entropy_reg_loss

            proto_sep_loss = prototype_separation_loss(prototypes=prototypes, temperature=args.temp_logits, device=device)
            cluster_loss += args.weight_proto_sep * proto_sep_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, temperature=args.temp_unsup_con)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = F.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss(temperature=args.temp_sup_con)(student_proj, labels=sup_con_labels)

            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'distill_loss: {distill_loss.item():.4f} '
            pstr += f'entropy_reg_loss: {entropy_reg_loss.item():.4f} '
            pstr += f'proto_sep_loss: {proto_sep_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss = 0
            loss += (1 - args.weight_sup) * cluster_loss + args.weight_sup * cls_loss
            loss += (1 - args.weight_sup) * contrastive_loss + args.weight_sup * sup_con_loss

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        #if new_acc_test > best_test_acc_ubl:
        #if old_acc_test > best_test_acc_lab and epoch > 100:
        if all_acc > best_train_acc_all:

            #args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            args.logger.info(f'Best ACC on all Classes on train set: {all_acc:.4f}...')
            args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            # inductive
            #best_test_acc_ubl = new_acc_test
            best_test_acc_lab = old_acc_test
            # transductive
            best_train_acc_lab = old_acc
            best_train_acc_ubl = new_acc
            best_train_acc_all = all_acc

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits, _ = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--init_prototypes', action='store_true', default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--weight_sup', type=float, default=0.35)
    parser.add_argument('--weight_entropy_reg', type=float, default=2)
    parser.add_argument('--weight_proto_sep', type=float, default=1)

    parser.add_argument('--temp_logits', default=0.1, type=float, help='cosine similarity of prototypes to classification logits temperature')
    parser.add_argument('--temp_teacher_logits', default=0.05, type=float, help='sharpened logits temperature of teacher')
    #parser.add_argument('--temp_proto_sep', default=0.1, type=float, help='prototype separation temperature')
    parser.add_argument('--temp_sup_con', default=0.07, type=float, help='supervised contrastive loss temperature')
    parser.add_argument('--temp_unsup_con', default=1.0, type=float, help='unsupervised contrastive loss temperature')

    parser.add_argument('--wait_ratio_epochs', default=0, type=int, help='Number of warmup epochs for the confidence filter.')
    parser.add_argument('--ramp_ratio_teacher_epochs', default=100, type=int, help='Number of warmup epochs for the confidence filter.')

    parser.add_argument('--init_ratio', default=0.2, type=float, help='initial confidence filter ratio')
    parser.add_argument('--final_ratio', default=1.0, type=float, help='final confidence filter ratio')

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.exp_root = 'dev_outputs_fix'

    init_experiment(args, runner_name=['ProtoGCD'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True


    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
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
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # --------------------
    # Initialize prototypes
    # --------------------
    prototypes_init = None
    if args.init_prototypes:
        prototype_init_path = './init_prototypes/%s_prototypes.pt' % args.dataset_name
        print('load initialized prototypes from: %s' % prototype_init_path)
        prototypes_init = torch.load(prototype_init_path)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers,
                         init_prototypes=prototypes_init, num_labeled_classes=args.num_labeled_classes)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
