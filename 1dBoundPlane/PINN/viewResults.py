import torch
import matplotlib.pyplot as plt
import numpy as np


def ExactPhi(x, L, D, S, a):
    return (
        S
        * L
        / (2 * D)
        * (np.exp(-np.abs(x) / L) - np.exp(-(a - np.abs(x)) / L))
        / (1 + np.exp(-a / L))
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 载入模型
model = torch.load("runs/run_1/model.pth")
Ntest = 5000
L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
S = 1
a = 2.0
x = np.linspace(-a / 2, a / 2, Ntest).reshape((Ntest, 1))
# plot loss
# 导入losses.txt等
path = "runs/run_2"
losses = np.loadtxt("%s/losses.txt" % path)
losses_r = np.loadtxt("%s/losses_f.txt" % path)
losses_bc1 = np.loadtxt("%s/losses_b1.txt" % path)
losses_bc2 = np.loadtxt("%s/losses_b2.txt" % path)
fig = plt.figure()
ax = fig.gca()
# 设置 y 轴为对数刻度
plt.yscale("log")
plt.plot(losses, label="loss")
plt.plot(losses_r, label="loss_r")
plt.plot(losses_bc1, label="loss_bc1")
# plt.plot(losses_bc2, label="loss_bc2")
plt.legend()
plt.savefig("losses.pdf")
