import os
import logging
import datetime
import torch

from config import config, overwrite_config_with_args, dump_config
from read_data import read_data, read_quad
from data_utils import heads_tails, inplace_shuffle, batch_by_num
from TTransE import TTransE
from logger_init import logger_init
from corrupter import BernCorrupterMulti
import numpy
from sklearn.preprocessing import MinMaxScaler

logger_init()
torch.cuda.set_device("cuda:0")
overwrite_config_with_args()
dump_config()


class MultiVarLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiVarLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def get_total_number(inPath):
    with open(inPath, 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


dataset_name = 'YAGO'
model_g = 'TTransE'
model_d = 'TATransE'
config().task.dir = dataset_name
task_dir = config().task.dir
n_ent, n_rel, n_time = get_total_number(os.path.join(task_dir, 'stat.txt'))

models = {'TTransE': TTransE}
config().g_config = model_g
config().d_config = model_d
gen_config = config()[config().g_config]
dis_config = config()[config().d_config]
gen = models[config().g_config](n_ent, n_rel, n_time, gen_config)
dis = models[config().d_config](n_ent, n_rel, n_time, dis_config)
gen.load(os.path.join(task_dir, gen_config.model_file))
# dis.load(os.path.join(task_dir, dis_config.model_file))

train_data = read_data(os.path.join(task_dir, 'train.txt'))
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'))
test_data = read_data(os.path.join(task_dir, 'test.txt'))
filt_heads, filt_tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

dis.test_link(test_data, n_ent, filt_heads, filt_tails)

corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, config().adv.n_sample)
src, rel, dst, time = train_data
n_train = len(src)
n_epoch = config().adv.n_epoch
n_batch = config().adv.n_batch
mdl_name = config().g_config + '_' + config().d_config + '_gan_dis_' + datetime.datetime.now().strftime(
    "%m%d%H%M%S") + '.mdl'
print('Before train, model path: ', mdl_name)
best_perf = 0
avg_reward = 0

# LSTM part
train_quad_lstm, train_time_lstm = read_quad(os.path.join(task_dir, 'train.txt'))

alltime_o_train = {}
alltime_s_train = {}
for i in range(len(train_time_lstm)):
    alltime_o_train[train_time_lstm[i]] = []
    alltime_s_train[train_time_lstm[i]] = []

for i in range(len(train_quad_lstm)):
    s, p, o, t = train_quad_lstm[i]
    alltime_o_train[t].append(o)
    alltime_s_train[t].append(s)

alltime_quantity_o_train = {}
alltime_quantity_s_train = {}
for i in range(len(train_time_lstm)):
    alltime_quantity_o_train[train_time_lstm[i]] = [0 for _ in range(n_ent)]
    alltime_quantity_s_train[train_time_lstm[i]] = [0 for _ in range(n_ent)]
for i in range(len(train_time_lstm)):
    for o in alltime_o_train[train_time_lstm[i]]:
        alltime_quantity_o_train[train_time_lstm[i]][o] += 1
    for s in alltime_s_train[train_time_lstm[i]]:
        alltime_quantity_s_train[train_time_lstm[i]][s] += 1

lstm_model_name = 'lstm_model_d_' + dataset_name + '.pth'
_, _, _, _, _, look_back = lstm_model_name.split('_')
look_back = int(look_back)
lstm_model = torch.load(lstm_model_name)
lstm_model.eval()

time_steps = len(train_time_lstm)
num_features = n_ent
data = []
for time_idx in train_time_lstm:
    data.append(alltime_quantity_o_train[time_idx])
data = numpy.array(data)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

future_steps = 5
last_known = data_scaled[-look_back:, :]
future_predictions = []
last_known_tensor = torch.tensor([last_known], dtype=torch.float32)

for _ in range(future_steps):
    last_known_tensor = last_known_tensor[:, -10:, ]
    with torch.no_grad():
        future_pred = lstm_model(last_known_tensor)
        # print(future_pred)
    future_predictions.append(future_pred.numpy().flatten())

    future_pred = future_pred.unsqueeze(0)
    # future_pred = torch.tensor(list(future_pred), dtype=torch.float32)
    last_known_tensor = torch.cat((last_known_tensor, future_pred), dim=1)


future_predictions = scaler.inverse_transform(numpy.array(future_predictions).reshape(-1, num_features))

for i1 in range(len(future_predictions)):
    for enti in range(len(future_predictions[i1])):
        future_predictions[i1][enti] = round(future_predictions[i1][enti])

future_predictions = list(future_predictions)


### TKGAN train
for epoch in range(n_epoch):
    epoch_d_loss = 0
    epoch_reward = 0
    src_cand, rel_cand, dst_cand, time_cand = corrupter.corrupt(src, rel, dst, time, keep_truth=False)
    for s, r, t, tim, ss, rs, ts, tims in batch_by_num(n_batch, src, rel, dst, time, src_cand, rel_cand, dst_cand, time_cand, n_sample=n_train):
        gen_step = gen.gen_step(ss, rs, ts, tims, temperature=config().adv.temperature)
        src_smpl, dst_smpl = next(gen_step)
        losses, rewards = dis.dis_step(s, r, t, tim, src_smpl.squeeze(), dst_smpl.squeeze())
        epoch_reward += torch.sum(rewards)
        rewards = rewards - avg_reward
        gen_step.send(rewards.unsqueeze(1))
        epoch_d_loss += torch.sum(losses)
    avg_loss = epoch_d_loss / n_train
    avg_reward = epoch_reward / n_train
    logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
    if (epoch + 1) % config().adv.epoch_per_test == 0:
        perf = dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
        if perf > best_perf:
            best_perf = perf
            dis.save(os.path.join(config().task.dir, mdl_name))

name = 'TTransE_TATransE_gan_dis_1129150640.mdl'
dis.load(os.path.join(config().task.dir, name))
print(os.path.join(config().task.dir, name))
dis.test_link(test_data, n_ent, filt_heads, filt_tails)

### TTT train
dis.train_with_ttt(test_data, n_ent, filt_heads, filt_tails, future_predictions)
dis.test_link(test_data, n_ent, filt_heads, filt_tails)
