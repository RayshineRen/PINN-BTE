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


class PINN(nn.Module):
    def __init__(self, layers, L, D, S, a, d, r_f, r_b, phi_b):
        super(PINN, self).__init__()
        self.dnn = NeuralNet(layers).to(device)
        self.adam_optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.max_iter_adam = 100000
        self.scheduler = StepLR(self.adam_optimizer, step_size=1000, gamma=0.95)
        self.bfgs_optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1e-2,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.optimizer = None
        self.iter = 0
        self.L = L
        self.D = D
        self.S = S
        self.a = a
        self.d = d
        self.r_f = r_f
        self.r_b = r_b
        self.phi_b = phi_b
        self.z_p = Variable(
            torch.from_numpy(np.array([[a + d]])).float(), requires_grad=True
        ).to(device)
        self.losses = []
        self.losses_f = []
        self.losses_b1 = []
        self.losses_b2 = []

    def f(self, r):
        phi = self.dnn(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0]
        return phi_rr - phi / self.L**2

    def bc(self, r):
        phi = self.dnn(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return -self.D * phi_r - self.S / 2.0

    def loss_func(self):
        self.optimizer.zero_grad()
        f_pred = self.f(self.r_f)
        # bc_pred = self.bc(self.r_b)
        bc_pred = self.dnn(self.r_b)
        loss_r = torch.mean(torch.square(f_pred))
        loss_bc1 = torch.mean(torch.square(bc_pred - self.phi_b))
        loss_bc2 = torch.mean(torch.square(self.dnn(self.z_p)))
        loss = loss_r + loss_bc1 + loss_bc2
        loss.backward()
        elapsed = time.process_time() - self.init_time
        if self.iter % 500 == 0:
            print(
                "Iter %d, Loss: %.4e, Loss_r: %.4e, Loss_bc1: %.4e, Loss_bc2: %.4e, Time: %.4f"
                % (
                    self.iter,
                    loss.item(),
                    loss_r.item(),
                    loss_bc1.item(),
                    loss_bc2.item(),
                    elapsed,
                )
            )
        self.losses.append(loss.item())
        self.losses_f.append(loss_r.item())
        self.losses_b1.append(loss_bc1.item())
        self.losses_b2.append(loss_bc2.item())
        return loss

    def train(self):
        self.dnn.train()
        self.optimizer = self.adam_optimizer
        self.init_time = time.process_time()
        while self.iter < self.max_iter_adam:
            self.optimizer.step(self.loss_func)
            self.iter += 1
            self.scheduler.step()
        print("Adam training finished!")
        print(
            "Iter %d, Loss: %.4e, Loss_r: %.4e, Loss_bc1: %.4e, Loss_bc2: %.4e, Time: %.4f"
            % (
                self.iter,
                self.losses[-1],
                self.losses_f[-1],
                self.losses_b1[-1],
                self.losses_b2[-1],
                time.process_time() - self.init_time,
            )
        )
        self.end_time = time.process_time()
        self.optimizer = self.bfgs_optimizer
        # self.optimizer.step(self.loss_func)

    def predict(self, r):
        self.dnn.eval()
        phi = self.dnn(r)
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


L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
S = 1
a = 2.0
Ntest = 10000
x = np.linspace(-a / 2, a / 2, Ntest).reshape((Ntest, 1))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "/home/ray/code/test/PDE/PINN_BTE/1dBoundPlane/PINN/runs/run_4"
model = torch.load("%s/model.pth" % (path))
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
phi_pred = model.predict(pt_x)
phi_exact = ExactPhi(x, L, D, S, a)
# 计算L2error
L2error = np.sqrt(np.mean(np.square(phi_exact - phi_pred)))
print("L2error: %.4e" % (L2error))
# plt.plot(x, phi_pred, "b-", linewidth=2, label="PINN")
# plt.plot(x, phi_exact, "r--", linewidth=2, label="Exact")
# plt.xlabel(r"$x$", fontsize=16)
# plt.ylabel(r"$\phi$", fontsize=16)
# plt.legend(loc="upper right", frameon=False, fontsize=16)
# plt.savefig("%s/phi.png" % (path))
