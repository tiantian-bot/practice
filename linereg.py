import numpy as np
import torch
from torch.utils import data
from d2l import torch as  d2l

# from mian import num_epochs


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.01, Y.shape)

    return X, Y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2

feature, label = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train):
    # print(*data_arrays)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((feature, label), batch_size, True)
# print(iter(data_iter))

next(iter(data_iter))
# print(next(iter(data_iter)))

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


loss = nn.MSELoss()


trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 5

for epoch in range(num_epochs):
    for X, y in data_iter:
        # print(X)
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()


    l = loss(net(feature), label)
    print(epoch, l.item())

