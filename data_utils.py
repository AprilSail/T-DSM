from random import randint
from collections import defaultdict
import torch

def heads_tails(n_ent, train_data, valid_data=None, test_data=None):
    train_src, train_rel, train_dst, train_time = train_data
    if valid_data:
        valid_src, valid_rel, valid_dst, valid_time = valid_data
    else:
        valid_src = valid_rel = valid_dst = valid_time = []
    if test_data:
        test_src, test_rel, test_dst, test_time = test_data
    else:
        test_src = test_rel = test_dst = test_time = []
    all_src = train_src + valid_src + test_src
    all_rel = train_rel + valid_rel + test_rel
    all_dst = train_dst + valid_dst + test_dst
    all_time = train_time + valid_time + test_time
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for s, r, t, tim in zip(all_src, all_rel, all_dst, all_time):
        tails[(s, r, tim)].add(t)
        heads[(t, r, tim)].add(s)
    heads_sp = {}
    tails_sp = {}
    for k in tails.keys():
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                               torch.ones(len(tails[k])), torch.Size([n_ent]))
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                               torch.ones(len(heads[k])), torch.Size([n_ent]))
    return heads_sp, tails_sp


def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(randint(0, i))
    for ls in lists:
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]


def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    for i in range(n_batch):
        head = int(n_sample * i / n_batch)
        tail = int(n_sample * (i + 1) / n_batch)
        ret = [ls[head:tail] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]