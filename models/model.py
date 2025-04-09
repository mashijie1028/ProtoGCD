import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, init_prototypes=None,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, num_labeled_classes=50):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # prototypes
        self.prototype_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.prototype_layer.weight_g.data.fill_(1)
        self.prototype_layer.weight_g.requires_grad = False
        print('prototype size: ', self.prototype_layer.weight_v.size())

        if init_prototypes is not None:
            print('initialize templates with labeled means and k-means centroids...')
            print(init_prototypes.size())
            print(init_prototypes)
            #self.prototype_layer.weight_v.data.copy_(init_prototypes)
            self.prototype_layer.weight_v.data[:num_labeled_classes].copy_(init_prototypes[:num_labeled_classes])
            print(self.prototype_layer.weight_v)
        else:
            print('randomly initialize prototypes...')
            print(self.prototype_layer.weight_v)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.prototype_layer(x)

        prototypes = self.prototype_layer.weight_v.clone()
        normed_prototypes = F.normalize(prototypes, dim=-1, p=2)

        return x_proj, logits, normed_prototypes



class DINOHead_k(nn.Module):
    '''
    DINOHead for estimating k.
    difference with DINOHead: `forward()`, return one more `x`
    date: 20230515
    '''
    def __init__(self, in_dim, out_dim, use_bn=False, init_prototypes=None,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, num_labeled_classes=50):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # prototypes
        self.prototype_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.prototype_layer.weight_g.data.fill_(1)
        self.prototype_layer.weight_g.requires_grad = False
        print('prototype size: ', self.prototype_layer.weight_v.size())

        if init_prototypes is not None:
            print('initialize templates with labeled means and k-means centroids...')
            print(init_prototypes.size())
            print(init_prototypes)
            #self.prototype_layer.weight_v.data.copy_(init_prototypes)
            self.prototype_layer.weight_v.data[:num_labeled_classes].copy_(init_prototypes[:num_labeled_classes])
            print(self.prototype_layer.weight_v)
        else:
            print('randomly initialize prototypes...')
            print(self.prototype_layer.weight_v)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.prototype_layer(x)

        prototypes = self.prototype_layer.weight_v.clone()
        normed_prototypes = F.normalize(prototypes, dim=-1, p=2)

        return x, x_proj, logits, normed_prototypes



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
