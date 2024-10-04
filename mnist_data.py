import torch
from IPython import display
from d2l import torch as d2l
from matplotlib import pyplot as plt

import os

# from toolz.curried import peekn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#定义一下小批量个数为256,取256张图片
batch_size = 256
#读取load_data_fashion_mnist中的数据，然后划分训练集和测试集,都取256张
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# train_iter分为图片和真实值
# print(next(iter(train_iter))[0][0].shape, next(iter(train_iter))[1].shape)

# 原本的图片是 28乘28一共有784个x 想象成一行方程 w总共有784个 然后还有一个b
num_inputs = 784
# 输出的label刚好有10种
num_outputs = 10

# W （784,10）
# b 10 一维
# W和b都要对梯度更新
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 最后表示10行 785列的一个方程组

# print(W.shape)
# print(b.shape)

#softmax让所有数变成正数，有负数会影响判断
#softmax回归的输出值个数等于标签里的类别数，这里输出完得到0-1的一个数（向量）
def softmax(X):
    #所有来个e，使得X_exp中的数为正数
    X_exp = torch.exp(X)
    #使得输出0-1取值     keepdim 保持输入输出维度一致
    partition = X_exp.sum(1, keepdim=True)
    #这里的广播机制是因为X_exp  partition的维度不一致
    return X_exp / partition  # 这里应用了广播机制


# #展现一下softmax实际的效果
# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))


# matmul代表的是矩阵之间的乘法 不是点乘
# X.reshape 256,784 W 784,10
# 这里就开始计算输出，后面用softmax使其成为0-1
def net(X):
    # X 256,1,28,28
    # X.reshape((-1, W.shape[0])) (256,784) 将4维向量变为2维向量
    # (torch.matmul(X.reshape((-1, W.shape[0])), W) (256,10)
    # 后面的b其实是(1，10) 这里运用广播机制 b变成（256，10）
    # 最后利用softmax 得到256行 10列  得到不同照片10的标签分别的置信度！！！！！！！！！
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# #这个表明预测的值
# #前面[]表示第几个[]；后面的表示[]的第几个
# print(y_hat[[0, 1], y])


# 因为是分类问题，所以我们要使用交叉熵 -sum(ylog(y_hat))
# y_hat[range(len(y_hat))表明行数
# 分类问题只有一个交叉熵可用，也就是y=1的交叉熵！！！！！！y=0的时候可以自己忽略
def cross_entropy(y_hat, y):
    # 这里的return返回的是-log预测标签的值，真实值逼近预测值！！
    # 这里-log保证交叉熵永远是正数
    return - torch.log(y_hat[range(len(y_hat)), y])


# print(cross_entropy(y_hat, y))


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    # len(y_hat.shape)表示y_hat的行数  y_hat.shape[1]表示y_hat的列数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 经过softmax以后 10个预测值当中取最大值的下标存在y_hat中
        # argmax(axis=1)取每行中最大的那个下标
        # y_hat.type(y.dtype)将y_hat的类型转换成跟y一样的
        # y_hat.type(y.dtype) == y 转换成布尔值
        y_hat = y_hat.argmax(axis=1)
        # print(y_hat)
        # print(y_hat.type(y.dtype))
        # print(y)
    cmp = y_hat.type(y.dtype) == y
    # float将布尔值转换为浮点数
    # print(cmp.type(y.dtype).sum())
    return float(cmp.type(y.dtype).sum())

# print(torch.tensor([True, True, True, True, False]).type(torch.tensor([0, 0, 0, 0, 0]).dtype))
# print(accuracy(y_hat, y) / len(y))

# 这个代码看一看__init__和add就好了
# 这个是[[],[]]自己创了一个向量
class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n
        # print(self.data)

    # *args表示传入一个数组 zip表示((0,0),(1,2))然后变为 (0,1) (0,2)
    def add(self, *args):
        # print(args)
        # for m, n in zip(self.data, args):
        #     print(m)
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # print(self.data)

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#用来评估在任意模型nel的准确率
#这里传入实际用的模型和数据迭代器，计算一下数据在模型上的精度
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    #如果使用torch.nn.Module形成的模型，if是true进入下面的语句
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，评估模式不用计算梯度
    #这边定义了一个metric,初始化为[0.0, 0.0]
    metric = Accumulator(2)  # 正确预测数、预测总数

    # print(metric.data)

    with torch.no_grad():
        for X, y in data_iter:
            #net(X)算出0-1的预测值
            #accuracy()得到正确分类的样本数，y.numel()样本总数
            metric.add(accuracy(net(X), y), y.numel())
            # print(metric.data)
    #return 返回正确分类样本数的比率
    return metric[0] / metric[1]

#我们这里的net就是softmax得到的一些置信度，test_iter是测试集的一些数据
#显示的是正确分类的比例
# print(evaluate_accuracy(net, test_iter))
# 传入定义好的模型（手动定义或者torch.nn）、训练集、损失函数(交叉熵)、
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    # torch.nn神经网络
    if isinstance(net, torch.nn.Module):
        # 开始训练,计算梯度
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    # 长度为3的累加器
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数

        # y_hat经过softmat的置信度
        y_hat = net(X)
        l = loss(y_hat, y)
        # 如果updater是torch.optim.Optimizer的话
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            # 梯度设为0
            updater.zero_grad()
            # 计算梯度
            l.mean().backward()
            # 更新一下
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            # 向量转换为标量
            l.sum().backward()
            # 根据batch_size大小更新一下
            updater(X.shape[0])
        # 返回cost fuction，精确度，样本总数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
# 动画这里有空去掌握一下，这里不是重点


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.01)
        display.clear_output(wait=True)

# 这个是用来训练本次训练集的模型
# 传入模型、训练集、测试集、损失函数、迭代次数、updater（sgd 小批量梯度下降）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    # 可视化的套件忽略
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 返回训练损失和训练精度
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # print(train_metrics)
        # 返回测试集的精度
        test_acc = evaluate_accuracy(net, test_iter)
        # print((test_acc,))
        # animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1


# 小批量随机梯度下降优化损失函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 20
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=20):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # 这里的reshape还是有点没懂
    # print(titles)
    # print('===================')
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter)