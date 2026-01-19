import os
import logging
import torch
from corrupter import BernCorrupter, BernCorrupterMulti
from read_data import read_data
from config import config, overwrite_config_with_args, dump_config
from logger_init import logger_init
from data_utils import inplace_shuffle, heads_tails
from TTransE import TTransE

def get_total_number(inPath):
    with open(inPath, 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])

logger_init()
torch.cuda.set_device("cuda:0")
overwrite_config_with_args()
dataset_name ='YAGO'
model_name = 'TTransE'
config().task.dir = dataset_name
task_dir = config().task.dir
n_ent, n_rel, n_time = get_total_number(os.path.join(task_dir, 'stat.txt'))

train_data = read_data(os.path.join(task_dir, 'train.txt'))
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'))
test_data = read_data(os.path.join(task_dir, 'test.txt'))
heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: gen.test_link(valid_data, n_ent, heads, tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

config().pretrain_config = model_name
mdl_type = config().pretrain_config
gen_config = config()[mdl_type]

if mdl_type == 'TTransE':
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    gen = TTransE(n_ent, n_rel, n_time, gen_config)

gen.pretrain(train_data, corrupter, tester)
gen.load(os.path.join(task_dir, gen_config.model_file))
gen.test_link(test_data, n_ent, heads, tails)
