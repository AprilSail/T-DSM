import torch as t

def mrr_mr_hitk(scores, target, k=10):
    _, sorted_idx = t.sort(scores)
    find_target = sorted_idx == target
    # print(find_target)
    if True in find_target:
        target_rank = t.nonzero(find_target)[0, 0] + 1
    elif True not in find_target:
        target_rank = 999999
    return 1 / target_rank, target_rank, int(target_rank <= 1), int(target_rank <= 3), int(target_rank <= 10)
