import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def entropy_regularization_loss(logits, temperature):
    avg_probs = (logits / temperature).softmax(dim=1).mean(dim=0)
    entropy_reg_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
    return entropy_reg_loss


def prototype_separation_loss(prototypes, temperature=0.1, base_temperature=0.1, device='cuda'):
    num_classes = prototypes.size(0)
    labels = torch.arange(0, num_classes).to(device)
    labels = labels.contiguous().view(-1, 1)

    mask = (1- torch.eq(labels, labels.T).float()).cuda()

    logits = torch.div(torch.matmul(prototypes, prototypes.T), temperature)

    mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
    mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]

    # loss
    loss = temperature / base_temperature * mean_prob_neg.mean()

    return loss



class DistillLoss_ratio(nn.Module):
    def __init__(self, num_classes=100, wait_ratio_epochs=0, ramp_ratio_teacher_epochs=100,
                 nepochs=200, ncrops=2, init_ratio=0.0, final_ratio=1.0,
                 temp_logits=0.1, temp_teacher_logits=0.05, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.temp_logits = temp_logits
        self.temp_teacher_logits = temp_teacher_logits
        self.ncrops = ncrops
        self.ratio_schedule = np.concatenate((
            np.zeros(wait_ratio_epochs),
            np.linspace(init_ratio,
                        final_ratio, ramp_ratio_teacher_epochs),
            np.ones(nepochs - wait_ratio_epochs - ramp_ratio_teacher_epochs) * final_ratio
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.temp_logits
        student_out = student_out.chunk(self.ncrops)

        # confidence filtering
        ratio_epoch = self.ratio_schedule[epoch]
        teacher_out = F.softmax(teacher_output / self.temp_teacher_logits, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        teacher_label = []
        for i in range(self.ncrops):
            top2 = torch.topk(teacher_out[i], k=2, dim=-1, largest=True)[0]
            top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
            filter_number = int(len(teacher_out[i]) * ratio_epoch)
            topk_filter = torch.topk(top2_div, k=filter_number, largest=True)[1]
            pseudo_label = F.one_hot(teacher_out[i].argmax(dim=-1), num_classes=self.num_classes)
            pseudo_label = pseudo_label.float()
            teacher_out[i][topk_filter] = pseudo_label[topk_filter]
            teacher_label.append(teacher_out[i])

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_label):
            #for v in range(len(student_out)):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                #loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss



class DistillLoss_ratio_ramp(nn.Module):
    def __init__(self, num_classes=100, wait_ratio_epochs=0, ramp_ratio_teacher_epochs=100,
                 nepochs=200, ncrops=2, init_ratio=0.2, final_ratio=1.0,
                 temp_logits=0.1, temp_teacher_logits_init=0.07, temp_teacher_logits_final=0.04, ramp_temp_teacher_epochs=30,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.temp_logits = temp_logits
        # self.temp_teacher_logits_init = temp_teacher_logits_init
        # self.temp_teacher_logits_final = temp_teacher_logits_final
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(temp_teacher_logits_init,
                        temp_teacher_logits_final, ramp_temp_teacher_epochs),
            np.ones(nepochs - ramp_temp_teacher_epochs) * temp_teacher_logits_final
        ))
        self.ratio_schedule = np.concatenate((
            np.zeros(wait_ratio_epochs),
            np.linspace(init_ratio,
                        final_ratio, ramp_ratio_teacher_epochs),
            np.ones(nepochs - wait_ratio_epochs - ramp_ratio_teacher_epochs) * final_ratio
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.temp_logits
        student_out = student_out.chunk(self.ncrops)

        # confidence filtering
        temp_teacher_epoch = self.teacher_temp_schedule[epoch]
        ratio_epoch = self.ratio_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp_teacher_epoch, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        teacher_label = []
        for i in range(self.ncrops):
            top2 = torch.topk(teacher_out[i], k=2, dim=-1, largest=True)[0]
            top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
            filter_number = int(len(teacher_out[i]) * ratio_epoch)
            topk_filter = torch.topk(top2_div, k=filter_number, largest=True)[1]
            pseudo_label = F.one_hot(teacher_out[i].argmax(dim=-1), num_classes=self.num_classes)
            pseudo_label = pseudo_label.float()
            teacher_out[i][topk_filter] = pseudo_label[topk_filter]
            teacher_label.append(teacher_out[i])

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_label):
            #for v in range(len(student_out)):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                #loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
