import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import datetime


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName):
    interval = 1
    if 'GDELT' in inPath:
        interval = 15
    if 'ICEWS18' in inPath or 'ICEWS14' in inPath:
        interval = 24
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(int(line_split[3]) / interval)
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, :])
    return np.array(X), np.array(Y)


class MultiVarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiVarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


dataset_name = 'YAGO'
look_back = 20
pred_type = 'O'
hidden_size = 1000
num_layers = 2
epochs = 1000
model_file = 'lstm_model_d_' + dataset_name + '_lb_' + str(look_back) + '_SorO_' + str(pred_type) + '_hs_' + str(
    hidden_size) + '_nl_' + str(num_layers) + '_e_' + str(epochs) + '_' + datetime.datetime.now().strftime(
    "%m%d%H%M%S") + '.pth'
print('Train model: ', model_file)

inpath = './' + dataset_name + '/'
num_ent, num_rel = get_total_number(inpath, 'stat.txt')
train_data, train_times = load_quadruples(inpath, 'train.txt')
test_data, test_times = load_quadruples(inpath, 'test.txt')
dev_data, dev_times = load_quadruples(inpath, 'valid.txt')

alltime_o_train = {}
alltime_s_train = {}
for i in range(len(train_times)):
    alltime_o_train[train_times[i]] = []
    alltime_s_train[train_times[i]] = []

for i in range(len(train_data)):
    s, p, o, t = train_data[i]
    alltime_o_train[t].append(o)
    alltime_s_train[t].append(s)

alltime_quantity_train = {}
for i in range(len(train_times)):
    alltime_quantity_train[train_times[i]] = [0 for _ in range(num_ent)]
if pred_type == 'O':
    for i in range(len(train_times)):
        for o in alltime_o_train[train_times[i]]:
            alltime_quantity_train[train_times[i]][o] += 1
elif pred_type == 'S':
    for i in range(len(train_times)):
        for s in alltime_s_train[train_times[i]]:
            alltime_quantity_train[train_times[i]][s] += 1

time_steps = len(train_times)
num_features = num_ent
data = []
for time_idx in train_times:
    data.append(alltime_quantity_train[time_idx])
data = np.array(data)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

X, Y = create_dataset(data_scaled, look_back)

train_size = int(len(X) * 0.95)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

input_size = num_features
output_size = num_features
model = MultiVarLSTM(input_size, hidden_size, num_layers, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epochs):
    model.zero_grad()
    output = model(X_train)
    loss = loss_function(output, Y_train)
    loss.backward()
    optimizer.step()
    if (i + 1) % 5 == 0:
        print(f'Epoch [{i + 1}/{epochs}], Loss: {loss.item():.7f}')

model.eval()
with torch.no_grad():
    train_predict = model(X_train).numpy()
    test_predict = model(X_test).numpy()

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = scaler.inverse_transform(Y_train.numpy())
Y_test = scaler.inverse_transform(Y_test.numpy())

train_score = mean_squared_error(Y_train, train_predict)
test_score = mean_squared_error(Y_test, test_predict)
print(f'Train Score: {train_score:.7f} RMSE')
print(f'Test Score: {test_score:.7f} RMSE')

torch.save(model, model_file)
