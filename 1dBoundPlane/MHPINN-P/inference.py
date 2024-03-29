import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.layers = layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 2)]
        )
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, r):
        x = r
        for hidden_layer in self.hidden_layers:
            x = F.tanh(hidden_layer(x))
        y = self.output_layer(x)
        return y


class MHPINN(nn.Module):
    def __init__(self, head_nums, layers, L, D, a, d, r_s_f, r_s_phi_b, z_p_s):
        super(MHPINN, self).__init__()
        self.head_nums = head_nums
        self.neural_nets = nn.ModuleList(
            [NeuralNet(layers).to(device) for _ in range(self.head_nums)]
        )
        self.end_net = NeuralNet([layers[-1] * head_nums, 1]).to(device)
        self.L = L
        self.D = D
        self.a = a
        self.d = d
        self.r_s_f = r_s_f
        self.r_s_phi_b = r_s_phi_b
        self.z_p_s = z_p_s
        self.losses = []
        self.losses_r = []
        self.losses_bc1 = []
        self.losses_bc2 = []

    def forward(self, r):
        y = torch.cat([net(r) for net in self.neural_nets], dim=1)
        y = self.end_net(y)
        return y

    def f(self, r):
        phi = self.forward(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0]
        return phi_rr - phi / self.L**2

    def bc(self, r):
        phi = self.forward(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return -self.D * phi_r - r[:, 1] / 2.0

    def loss_func(self):
        f_pred = self.f(self.r_s_f)
        # bc_pred = self.bc(self.r_b)
        bc_pred = self.forward(self.r_s_phi_b[:, :2])
        self.loss_r = torch.mean(torch.square(f_pred))
        self.loss_bc1 = torch.mean(torch.square(bc_pred - self.r_s_phi_b[:, 2]))
        self.loss_bc2 = torch.mean(torch.square(self.forward(self.z_p_s)))
        self.loss = self.loss_r + self.loss_bc1 + self.loss_bc2
        return self.loss

    def predict(self, r):
        phi = self.forward(r)
        # phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return phi.detach().cpu().numpy()  # , phi_r.detach().cpu().numpy()


def ExactPhi(x, L, D, S, a):
    return (
        S
        * L
        / (2 * D)
        * (np.exp(-np.abs(x) / L) - np.exp(-(a - np.abs(x)) / L))
        / (1 + np.exp(-a / L))
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "/home/ray/code/test/PDE/PINN_BTE/1dBoundPlane/MH-PINN-P/runs/run_6"
L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
a = 2.0
Ntest = 10000
for s in [1, 5, 10, 15, 20]:
    x = np.linspace(-a / 2, a / 2, Ntest).reshape((Ntest, 1))
    s1 = np.ones((Ntest, 1)) * s
    x_s1 = np.concatenate((x, s1), axis=1)
    model = torch.load("%s/model.pth" % (path))
    pt_x_s1 = Variable(torch.from_numpy(x_s1).float(), requires_grad=True).to(device)
    phi_pred = model.predict(pt_x_s1)
    phi_exact = ExactPhi(x, L, D, s, a)
    # 计算L2error
    L2error = np.sqrt(np.mean(np.square(phi_exact - phi_pred)))
    print("L2error: %.4e" % (L2error))
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, phi_pred, "b-", linewidth=2, label="PINN")
    plt.plot(x, phi_exact, "r--", linewidth=2, label="Exact")
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$\phi$", fontsize=16)
    plt.legend(loc="upper right", frameon=False, fontsize=16)
    plt.savefig("%s/phi_s%d.png" % (path, s))
