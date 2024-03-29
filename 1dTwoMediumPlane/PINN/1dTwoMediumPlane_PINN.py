import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)


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
    def __init__(
        self, layers, L1, D1, L2, D2, S, a, r_f1, r_b1, phi_b1, r_f2, r_b2, phi_b2
    ):
        super(PINN, self).__init__()
        self.dnn = NeuralNet(layers).to(device)
        self.L1 = L1
        self.L2 = L2
        self.D1 = D1
        self.D2 = D2
        self.S = S
        self.a = a
        self.r_f1 = r_f1
        self.r_b1 = r_b1
        self.phi_b1 = phi_b1
        self.r_f2 = r_f2
        self.r_b2 = r_b2
        self.phi_b2 = phi_b2
        self.losses = []
        self.losses_r1 = []
        self.losses_r2 = []
        self.losses_bc1 = []
        self.losses_bc2 = []

    def forward(self, r):
        y = self.dnn(r)
        return y

    def f(self, r, L):
        phi = self.forward(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0]
        return phi_rr - phi / L**2

    def bc(self, r):
        phi = self.forward(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return -self.D * phi_r - self.S / 2.0

    def loss_func(self):
        f_pred1 = self.f(self.r_f1, self.L1)
        f_pred2 = self.f(self.r_f2, self.L2)
        # bc_pred = self.bc(self.r_b)
        bc_pred1 = self.forward(self.r_b1)
        bc_pred2 = self.forward(self.r_b2)
        self.loss_r1 = torch.mean(torch.square(f_pred1))
        self.loss_bc1 = torch.mean(torch.square(bc_pred1 - self.phi_b1))
        self.loss_r2 = torch.mean(torch.square(f_pred2))
        self.loss_bc2 = torch.mean(torch.square(bc_pred2 - self.phi_b2))
        self.loss = self.loss_r1 + self.loss_bc1 + self.loss_r2 + self.loss_bc2
        return self.loss

    def predict(self, r):
        phi = self.forward(r)
        # phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return phi.detach().cpu().numpy()  # , phi_r.detach().cpu().numpy()


def ExactPhi1(x, L1, D1, L2, D2, S, a):
    A1 = -S * L1 / (2 * D1)
    C1 = (
        S
        * L1
        / (2 * D1)
        * (D1 * L2 * np.cosh(a / (2 * L1)) + D2 * L1 * np.sinh(a / (2 * L1)))
        / (D2 * L1 * np.cosh(a / (2 * L1)) + D1 * L2 * np.sinh(a / (2 * L1)))
    )
    phi = A1 * np.sinh(np.abs(x) / L1) + C1 * np.cosh(np.abs(x) / L1)
    return phi


def ExactPhi2(x, L1, D1, L2, D2, S, a):
    A2 = (
        S
        * L1
        * L2
        / 2
        * np.exp(a / (2 * L2))
        / (D2 * L1 * np.cosh(a / (2 * L1)) + D1 * L2 * np.sinh(a / (2 * L1)))
    )
    phi = A2 * np.exp(-np.abs(x) / L2)
    return phi


# Medium 1
Nf1 = 1000
Nb1 = 100
L1 = np.sqrt(0.03)
lambda_tr1 = 0.0228
D1 = 1 / 3 * lambda_tr1
S = 1
a = 2.0
# Medium 2
Nf2 = 1000
Nb2 = 100
L2 = np.sqrt(0.08)
lambda_tr2 = 0.0456
D2 = 1 / 3 * lambda_tr2
# Medium 1 points
b_l_minus = -0.1
b_r_minus = -0.01
b_l_posit = +0.01
b_r_posit = +0.1
x_f_l = np.random.rand(Nf1 // 2) * (-a / 2 - b_l_minus) + b_l_minus
x_f_r = np.random.rand(Nf1 // 2) * (a / 2 - b_r_posit) + b_r_posit
x_f1 = np.concatenate((x_f_l, x_f_r), axis=0).reshape((Nf1, 1))
x_b_l = np.random.rand(Nb1 // 2) * (b_l_minus - b_r_minus) + b_r_minus
x_b_r = np.random.rand(Nb1 // 2) * (b_r_posit - b_l_posit) + b_l_posit
x_b1 = np.concatenate((x_b_l, x_b_r), axis=0).reshape((Nb1, 1))
phi_b1 = ExactPhi1(x_b1, L1, D1, L2, D2, S, a)
pt_x_f1 = Variable(torch.from_numpy(x_f1).float(), requires_grad=True).to(device)
pt_x_b1 = Variable(torch.from_numpy(x_b1).float(), requires_grad=True).to(device)
pt_phi_b1 = Variable(torch.from_numpy(phi_b1).float(), requires_grad=True).to(device)
# Medium 2 points
b_l_minus = -a / 2 - 0.1
b_r_minus = -a / 2 - 0.01
b_l_posit = a / 2 + 0.01
b_r_posit = a / 2 + 0.1
left_boundary = -a
right_boundary = a
x_f_l = np.random.rand(Nf2 // 2) * (left_boundary - b_l_minus) + b_l_minus
x_f_r = np.random.rand(Nf2 // 2) * (right_boundary - b_r_posit) + b_r_posit
x_f2 = np.concatenate((x_f_l, x_f_r), axis=0).reshape((Nf2, 1))
x_b_l = np.random.rand(Nb2 // 2) * (b_l_minus - b_r_minus) + b_r_minus
x_b_r = np.random.rand(Nb2 // 2) * (b_r_posit - b_l_posit) + b_l_posit
x_b2 = np.concatenate((x_b_l, x_b_r), axis=0).reshape((Nb2, 1))
phi_b2 = ExactPhi2(x_b2, L1, D1, L2, D2, S, a)
pt_x_f2 = Variable(torch.from_numpy(x_f2).float(), requires_grad=True).to(device)
pt_x_b2 = Variable(torch.from_numpy(x_b2).float(), requires_grad=True).to(device)
pt_phi_b2 = Variable(torch.from_numpy(phi_b2).float(), requires_grad=True).to(device)
layers = [1, 25, 25, 25, 25, 1]
model = PINN(
    layers,
    L1,
    D1,
    L2,
    D2,
    S,
    a,
    pt_x_f1,
    pt_x_b1,
    pt_phi_b1,
    pt_x_f2,
    pt_x_b2,
    pt_phi_b2,
).to(device)
model.apply(custom_weight_init)
# training
max_iters = 50000
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
start_time = time.process_time()

for epoch in range(1, max_iters + 1):
    optimizer.zero_grad()
    loss = model.loss_func()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.losses.append(loss.item())
    model.losses_r1.append(model.loss_r1.item())
    model.losses_r2.append(model.loss_r2.item())
    model.losses_bc1.append(model.loss_bc1.item())
    model.losses_bc2.append(model.loss_bc2.item())
    time_elapsed = time.process_time() - start_time
    if epoch % 500 == 0:
        print(
            "epoch: %d, loss: %.4e, loss_r1: %.4e, loss_bc1: %.4e, loss_r2: %.4e, loss_bc2: %.4e, time: %.4fs"
            % (
                epoch,
                loss.item(),
                model.loss_r1.item(),
                model.loss_bc1.item(),
                model.loss_r2.item(),
                model.loss_bc2.item(),
                time_elapsed,
            )
        )

# testing
# 检查runs文件夹内下标最大的子文件夹run_i，新建run_i+1
runs_folder = "runs"
subfolders = [
    f for f in os.listdir(runs_folder) if os.path.isdir(os.path.join(runs_folder, f))
]
max_index = 0
for folder in subfolders:
    try:
        folder_index = int(folder.split("_")[1])
        if folder_index > max_index:
            max_index = folder_index
    except ValueError:
        continue
new_folder_name = f"run_{max_index + 1}"
new_folder_path = os.path.join(runs_folder, new_folder_name)
os.makedirs(new_folder_path)

Ntest = 10000
x21 = np.linspace(-a, -a / 2, Ntest // 4).reshape((Ntest // 4, 1))
x22 = np.linspace(a / 2, a, Ntest // 4).reshape((Ntest // 4, 1))
x2 = np.concatenate((x21, x22), axis=0).reshape((Ntest // 2, 1))
fig = plt.figure()
ax = fig.gca()
Exact2 = ExactPhi2(x2, L1, D1, L2, D2, S, a)
plt.plot(x2, Exact2, label="Exact2")
x11 = np.linspace(-a / 2, 0, Ntest // 4).reshape((Ntest // 4, 1))
x12 = np.linspace(0, a / 2, Ntest // 4).reshape((Ntest // 4, 1))
x1 = np.concatenate((x11, x12), axis=0).reshape((Ntest // 2, 1))
Exact1 = ExactPhi1(x1, L1, D1, L2, D2, S, a)
plt.plot(x1, Exact1, label="Exact1")
plt.savefig("%s/Exact.pdf" % new_folder_path)

fig = plt.figure()
ax = fig.gca()
x = np.concatenate((x1, x2), axis=0).reshape((Ntest, 1))
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
u = model.predict(pt_x)
ms_u = u.reshape(pt_x.shape)
plt.plot(x, ms_u, label="PINN")
plt.savefig("%s/Pred.pdf" % new_folder_path)
# 计算相对L2-error
pt_x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True).to(device)
u1 = model.predict(pt_x1)
ms_u1 = u1.reshape(pt_x1.shape)
pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).to(device)
u2 = model.predict(pt_x2)
ms_u2 = u2.reshape(pt_x2.shape)
error1 = np.sqrt(np.square(ms_u1 - Exact1).sum()) / np.sqrt(np.square(Exact1).sum())
error2 = np.sqrt(np.square(ms_u2 - Exact2).sum()) / np.sqrt(np.square(Exact2).sum())
print("Error: %.4e" % ((error1 + error2) / 2))
# 写入结果文件
with open("%s/result.txt" % (new_folder_path), "a") as f:
    f.write("Nf1: %d\n" % Nf1)
    f.write("Nb1: %d\n" % Nb1)
    f.write("Nf2: %d\n" % Nf2)
    f.write("Nb2: %d\n" % Nb2)
    f.write("max_iters: %d\n" % max_iters)
    f.write("Error1: %.4e\n" % error1)
    f.write("Error2: %.4e\n" % error2)
    f.write("time: %.4fs\n" % (time_elapsed - start_time))
# 保存模型
torch.save(model, "%s/model.pth" % new_folder_path)
# 保存损失函数
np.savetxt(
    "%s/losses.txt" % new_folder_path,
    np.array(model.losses),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_f1.txt" % new_folder_path,
    np.array(model.losses_r1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_f2.txt" % new_folder_path,
    np.array(model.losses_r2),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_b1.txt" % new_folder_path,
    np.array(model.losses_bc1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_b2.txt" % new_folder_path,
    np.array(model.losses_bc2),
    fmt="%.8e",
    delimiter=",",
)
# 保存参数
np.savetxt(
    "%s/parameters.txt" % new_folder_path,
    np.array([L1, D1, L2, D2, S, a, layers[-2]]),
    fmt="%.8e",
    delimiter=",",
)
# 保存数据
np.savetxt(
    "%s/x_f1.txt" % new_folder_path,
    np.array(x_f1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/x_f2.txt" % new_folder_path,
    np.array(x_f2),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/x_b1.txt" % new_folder_path,
    np.array(x_b1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/x_b2.txt" % new_folder_path,
    np.array(x_b2),
    fmt="%.8e",
    delimiter=",",
)
