import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device(cuda0 if torch.cuda.is_available() else cpu)


class NeuralNet(nn.Module)
    def __init__(self, layers)
        super(NeuralNet, self).__init__()
        self.layers = layers
        self.input_layer = nn.Linear(layers[0], layers[1])
        self.activation = nn.Tanh()
        for i in range(1, len(layers) - 2)
            setattr(self, hidden_layer + str(i), nn.Linear(layers[i], layers[i + 1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, r)
        x = self.activation(self.input_layer(r))
        for i in range(1, len(self.layers) - 2)
            x = self.activation(getattr(self, hidden_layer + str(i))(x))
        y = self.output_layer(x)
        return y


class PINN
    def __init__(self, layers, L, D, S, r_f, r_b, phi_b)
        self.dnn = NeuralNet(layers).to(device)
        self.adam_optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.max_iter_adam = 50000
        self.scheduler = StepLR(self.adam_optimizer, step_size=1000, gamma=0.95)
        self.bfgs_optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1e-2,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0  np.finfo(float).eps,
            line_search_fn=strong_wolfe,
        )
        self.optimizer = None
        self.iter = 0
        self.L = L
        self.D = D
        self.S = S
        self.r_f = r_f
        self.r_b = r_b
        self.phi_b = phi_b
        self.losses = []
        self.losses_f = []
        self.losses_b = []

    def f(self, r)
        phi = self.dnn(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0]
        return phi_rr + 2.0  phi_r  r - phi  self.L2

    def bc(self, r)
        phi = self.dnn(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return -4  np.pi  self.D  r2  phi_r - self.S

    def loss_func(self)
        self.optimizer.zero_grad()
        f_pred = self.f(self.r_f)
        bc_pred = self.dnn(self.r_b)
        # bc_pred = self.bc(self.r_b)
        loss_r = torch.mean(torch.square(f_pred))
        loss_bc = torch.mean(torch.square(bc_pred - self.phi_b))
        # loss_bc = torch.mean(torch.square(bc_pred))
        loss = loss_r + loss_bc
        loss.backward()
        if self.iter % 500 == 0
            print(
                Iter %d, Loss %.4e, Loss_r %.4e, Loss_bc %.4e
                % (self.iter, loss.item(), loss_r.item(), loss_bc.item())
            )
        # print(
        #     Iter %d, Loss %.4e, Loss_r %.4e
        #     % (self.iter, loss.item(), loss_r.item())
        # )
        self.losses.append(loss.item())
        self.losses_f.append(loss_r.item())
        self.losses_b.append(loss_bc.item())
        return loss

    def train(self)
        self.dnn.train()
        self.optimizer = self.adam_optimizer
        while self.iter  self.max_iter_adam
            self.optimizer.step(self.loss_func)
            self.iter += 1
            self.scheduler.step()
        self.optimizer = self.bfgs_optimizer
        # self.optimizer.step(self.loss_func)

    def predict(self, r)
        self.dnn.eval()
        phi = self.dnn(r)
        # phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return phi.detach().cpu().numpy()  # , phi_r.detach().cpu().numpy()


def ExactPhi(r, L, D, S)
    return S  np.exp(-r  L)  (4  np.pi  D  r)


Nf = 5000
Nb = 1000
L = np.sqrt(0.03)
D = 0.04
S = 1.0
f_right = 1.0
b_left = 0.05
f_left = b_left + D
r_f = np.random.rand(Nf)  (f_right - f_left) + f_left
r_b = np.random.rand(Nb)  (f_left - b_left) + b_left
r_f = r_f.reshape((Nf, 1))
r_b = r_b.reshape((Nb, 1))
phi_b = ExactPhi(r_b, L, D, S)
pt_r_f = Variable(torch.from_numpy(r_f).float(), requires_grad=True).to(device)
pt_r_b = Variable(torch.from_numpy(r_b).float(), requires_grad=True).to(device)
pt_phi_b = Variable(torch.from_numpy(phi_b).float(), requires_grad=True).to(device)
model = PINN([1, 100, 100, 100, 100, 100, 1], L, D, S, pt_r_f, pt_r_b, pt_phi_b)
Dmodel = DataParallel(model, device_ids=[0, 7])
model.train()

Ntest = 5000
r = np.linspace(b_left, f_right, Ntest).reshape((Ntest, 1))
fig = plt.figure()
ax = fig.gca()
Exact = ExactPhi(r, L, D, S)
plt.plot(r, Exact, label=Exact)
plt.savefig(Exact.pdf)
with open(Exact.txt, w) as f
    for i in range(Exact.shape[0])
        f.write(str(r[i, 0]) +   + str(Exact[i, 0]) + n)

fig = plt.figure()
ax = fig.gca()
pt_r = Variable(torch.from_numpy(r).float(), requires_grad=True).to(device)
u = model.predict(pt_r)
ms_u = u.reshape(pt_r.shape)
plt.plot(r, ms_u, label=PINN)
plt.savefig(Pred.pdf)
