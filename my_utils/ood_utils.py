import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as sk
from tqdm import tqdm

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores_in(loader, model, args):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(loader)):

            images = images.cuda(non_blocking=True)
            feats, _, logits, prototypes = model(images)

            #output = model(data)
            #smax = to_np(F.softmax(logits, dim=1))
            smax = to_np(F.softmax(logits / args.temp_logits, dim=1))   # NOTE!!!
            output_ = to_np(logits)

            # if args.use_xent:
            #     _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            # else:
            #     _score.append(-np.max(smax, axis=1))

            if args.score == 'energy':
                _score.append(-to_np((args.T * torch.logsumexp(logits / args.T, dim=1))))
            elif args.score == 'mls':
                _score.append(-np.max(output_, axis=1))
            elif args.score == 'xent':
                #_score.append(to_np((logits.mean(1) - torch.logsumexp(logits, dim=1))))
                _score.append(to_np((logits.mean(1) / args.temp_logits - torch.logsumexp(logits / args.temp_logits, dim=1))))   # NOTE!!!
            elif args.score == 'proto':
                #top2 = np.topk(smax, k=2, dim=-1, largest=True)[0]
                #top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
                smax_sort = np.sort(smax, axis=1)
                top2_div = smax_sort[:, -1] / (smax_sort[:, -2] + 1e-6)
                _score.append(-top2_div)   # NOTE!!!
            elif args.score == 'margin':
                #top2 = np.topk(smax, k=2, dim=-1, largest=True)[0]
                #top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
                smax_sort = np.sort(smax, axis=1)
                top2_margin = smax_sort[:, -1] - smax_sort[:, -2]
                _score.append(-top2_margin)   # NOTE!!!
            else:
                _score.append(-np.max(smax, axis=1))

            preds = np.argmax(smax, axis=1)
            targets = labels.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            if args.score == 'xent':
                _right_score.append(to_np((logits.mean(1) / args.temp_logits - torch.logsumexp(logits / args.temp_logits, dim=1)))[right_indices])
                _wrong_score.append(to_np((logits.mean(1) / args.temp_logits - torch.logsumexp(logits / args.temp_logits, dim=1)))[wrong_indices])
            else:
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()


def get_ood_scores(loader, model, ood_num_examples, args):
    _score = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_size:
                break

            images = images.cuda(non_blocking=True)
            feats, _, logits, prototypes = model(images)

            #output = model(data)
            #smax = to_np(F.softmax(logits, dim=1))
            smax = to_np(F.softmax(logits / args.temp_logits, dim=1))   # NOTE!!!
            output_ = to_np(logits)

            # if args.use_xent:
            #     _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            # else:
            #     _score.append(-np.max(smax, axis=1))

            if args.score == 'energy':
                _score.append(-to_np((args.T * torch.logsumexp(logits / args.T, dim=1))))
            elif args.score == 'mls':
                _score.append(-np.max(output_, axis=1))
            elif args.score == 'xent':
                #_score.append(to_np((logits.mean(1) - torch.logsumexp(logits, dim=1))))
                _score.append(to_np((logits.mean(1) / args.temp_logits - torch.logsumexp(logits / args.temp_logits, dim=1))))   # NOTE!!!
            elif args.score == 'proto':
                #top2 = np.topk(smax, k=2, dim=-1, largest=True)[0]
                #top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
                smax_sort = np.sort(smax, axis=1)
                top2_div = smax_sort[:, -1] / (smax_sort[:, -2] + 1e-6)
                _score.append(-top2_div)   # NOTE!!!
            elif args.score == 'margin':
                #top2 = np.topk(smax, k=2, dim=-1, largest=True)[0]
                #top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
                smax_sort = np.sort(smax, axis=1)
                top2_margin = smax_sort[:, -1] - smax_sort[:, -2]
                _score.append(-top2_margin)   # NOTE!!!
            else:
                _score.append(-np.max(smax, axis=1))

    return concat(_score)[:ood_num_examples].copy()


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def print_measures(auroc, aupr_in, aupr_out, fpr_in, fpr_out, recall_level=recall_level_default):
    print('FPR(IN){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_in))
    print('FPR(OUT){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_out))
    print('AUROC: {:.2f}'.format(100 * auroc))
    print('AUPR(IN): {:.2f}'.format(100 * aupr_in))
    print('AUPR(OUT): {:.2f}'.format(100 * aupr_out))


def write_measures(auroc, aupr_in, aupr_out, fpr_in, fpr_out, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR(IN){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_in))
        f_log.write('\n')
        f_log.write('FPR(OUT){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_out))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f}'.format(100 * auroc))
        f_log.write('\n')
        f_log.write('AUPR(IN): {:.2f}'.format(100 * aupr_in))
        f_log.write('\n')
        f_log.write('AUPR(OUT): {:.2f}'.format(100 * aupr_out))
        f_log.write('\n')


def print_measures_with_std(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, recall_level=recall_level_default):
    print('FPR(IN){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_in), 100*np.std(fprs_in)))
    print('FPR(OUT){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_out), 100*np.std(fprs_out)))
    print('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
    print('AUPR(IN): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_in), 100*np.std(auprs_in)))
    print('AUPR(OUT): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_out), 100*np.std(auprs_out)))


def write_measures_with_std(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR(IN){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_in), 100*np.std(fprs_in)))
        f_log.write('\n')
        f_log.write('FPR(OUT){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_out), 100*np.std(fprs_out)))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
        f_log.write('\n')
        f_log.write('AUPR(IN): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_in), 100*np.std(auprs_in)))
        f_log.write('\n')
        f_log.write('AUPR(OUT): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_out), 100*np.std(auprs_out)))
        f_log.write('\n')
