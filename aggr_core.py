"""AGGR 使用的基础归一化与排名函数。"""

import torch
import torch.nn.functional as F


def l2norm(x, dim=-1):
    return F.normalize(x.float(), dim=dim, eps=1e-8)


def zscore_query(scores):
    x = scores.float()
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return torch.where(std > 1e-12, (x - mean) / std.clamp(min=1e-12), torch.zeros_like(x))


def rank_norm_from_scores(scores):
    bsz, k = scores.shape
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    base = torch.arange(k, device=scores.device).view(1, k).expand(bsz, -1)
    ranks.scatter_(1, order, base)
    return ranks.float() / max(k - 1, 1)


def base_rank_norm(base_ranks):
    k = base_ranks.shape[1]
    return (base_ranks.float() - 1.0) / max(k - 1, 1)
