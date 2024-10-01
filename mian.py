import  torch
import  random

from click.core import batch
from d2l import torch as d2l
from numba.core.cgutils import true_bit
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.01, Y.shape)

    return X, Y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2

feature, label = synthetic_data(true_w, true_b, 1000)

# print(feature)
# print(label)
# d2l.set_figsize()
# # print(feature[:, 1].detach().numpy())
# d2l.plt.scatter(feature[:, 1].detach().numpy(),
#                 label.detach().numpy(), 1)
# d2l.plt.show()

def data_iter(batch_size, feature, label):
    num_examples = len(label)
    indices = list(range(num_examples))
    random.shuffle(indices)
    # print(indices)
    for i in range(0, num_examples, batch_size):

        # print(i)
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])

        # print(batch_indices)

        yield  feature[batch_indices], label[batch_indices]

batch_size = 10

# for X, y in data_iter(batch_size, feature, label):
# #     pass
#     print(X, '\n', y, '\n')
    # break


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# print(w, b)
def linereg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 10
net = linereg
loss = squared_loss


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, feature, label):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(feature, w, b), label)
        print('epoch %d, train loss: %.6f' % (epoch, train_l.mean()))