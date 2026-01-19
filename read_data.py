import torch


def read_data(filename):
    src = []
    rel = []
    dst = []
    time = []
    with open(filename) as f:
        for ln in f:
            s, r, t, tim = ln.strip().split('\t')[:4]
            src.append(int(s))
            rel.append(int(r))
            dst.append(int(t))
            if 'ICEWS' in filename:
                time.append(int(int(tim)/24))
            elif 'GDELT' in filename:
                time.append(int(int(tim)/15))
            else:
                time.append(int(tim))
    return src, rel, dst, time


def read_quad(filename):
    quadg = []
    time = set()
    with open(filename) as f:
        for ln in f:
            s, r, t, tim = ln.strip().split('\t')[:4]
            if 'ICEWS' in filename:
                quadg.append([int(s), int(r), int(t), int(tim)/24])
                time.add(int(tim)/24)
            elif 'GDELT' in filename:
                quadg.append([int(s), int(r), int(t), int(tim)/15])
                time.add(int(tim)/15)
            else:
                quadg.append([int(s), int(r), int(t), int(tim)])
                time.add(int(tim))
    time = list(time)
    time.sort()
    return quadg, time
