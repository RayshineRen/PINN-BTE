import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


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


class ResNet(NeuralNet):
    def __init__(self, layers):
        super(ResNet, self).__init__(layers)

    def forward(self, r):
        x = self.hidden_layers[0](r)
        for i in range(1, len(self.hidden_layers)):
            x = F.tanh(self.hidden_layers[i](x)) + x
        y = self.output_layer(x)
        return y


class MHPINN(nn.Module):
    def __init__(self, head_nums, layers, L, D, a, d, r_s_f, r_s_phi_b, z_p_s):
        super(MHPINN, self).__init__()
        self.head_nums = head_nums
        self.neural_nets = nn.ModuleList(
            [ResNet(layers).to(device) for _ in range(self.head_nums)]
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
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0][:, 0].reshape(
            phi.shape
        )
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0][
            :, 0
        ].reshape(phi.shape)
        return phi_rr - phi / self.L**2

    def bc(self, r):
        phi = self.forward(r)
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0][:, 0].reshape(
            phi.shape
        )
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


Nf1 = 2000
Nf2 = 2000
Nf3 = 2000
Nf4 = 2000
Nf5 = 2000
Nf = Nf1 + Nf2 + Nf3 + Nf4 + Nf5
Nb1 = 1000
Nb2 = 1000
Nb3 = 1000
Nb4 = 1000
Nb5 = 1000
Nb = Nb1 + Nb2 + Nb3 + Nb4 + Nb5
L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
S1 = 1
S2 = 5
S3 = 10
S4 = 15
S5 = 20
a = 2.0
b_l_minus = -0.1
b_r_minus = -0.01
b_l_posit = +0.01
b_r_posit = +0.1
x_f_l = np.random.rand(Nf // 2) * (-a / 2 - b_l_minus) + b_l_minus
x_f_r = np.random.rand(Nf // 2) * (a / 2 - b_r_posit) + b_r_posit
x_f = np.concatenate((x_f_l, x_f_r), axis=0).reshape((Nf, 1))
# 随机取Nf1个点
x_f1 = x_f[np.random.choice(Nf, Nf1, replace=False)]
s_f1 = S1 * np.ones((Nf1, 1))
x_s_f1 = np.concatenate((x_f1, s_f1), axis=1)
# 再随机取Nf2个点
x_f2 = (np.setdiff1d(x_f, x_f1).reshape((Nf - Nf1, 1)))[
    np.random.choice(Nf - Nf1, Nf2, replace=False)
]
s_f2 = S2 * np.ones((Nf2, 1))
x_s_f2 = np.concatenate((x_f2, s_f2), axis=1)
# 再随机取Nf3个点
x_f3 = (
    np.setdiff1d(x_f, np.concatenate((x_f1, x_f2), axis=0)).reshape((Nf - Nf1 - Nf2, 1))
)[np.random.choice(Nf - Nf1 - Nf2, Nf3, replace=False)]
s_f3 = S3 * np.ones((Nf3, 1))
x_s_f3 = np.concatenate((x_f3, s_f3), axis=1)
# 再随机取Nf4个点
x_f4 = (
    np.setdiff1d(x_f, np.concatenate((x_f1, x_f2, x_f3), axis=0)).reshape(
        (Nf - Nf1 - Nf2 - Nf3, 1)
    )
)[np.random.choice(Nf - Nf1 - Nf2 - Nf3, Nf4, replace=False)]
s_f4 = S4 * np.ones((Nf4, 1))
x_s_f4 = np.concatenate((x_f4, s_f4), axis=1)
# 剩下的Nf5个点
x_f5 = np.setdiff1d(x_f, np.concatenate((x_f1, x_f2, x_f3, x_f4), axis=0)).reshape(
    (Nf - Nf1 - Nf2 - Nf3 - Nf4, 1)
)
s_f5 = S5 * np.ones((Nf5, 1))
x_s_f5 = np.concatenate((x_f5, s_f5), axis=1)
# 将所有的f点合并
x_s_f = np.concatenate((x_s_f1, x_s_f2, x_s_f3, x_s_f4, x_s_f5), axis=0)

x_b_l = np.random.rand(Nb // 2) * (b_l_minus - b_r_minus) + b_r_minus
x_b_r = np.random.rand(Nb // 2) * (b_r_posit - b_l_posit) + b_l_posit
x_b = np.concatenate((x_b_l, x_b_r), axis=0).reshape((Nb, 1))
# 随机取Nb1个点
x_b1 = x_b[np.random.choice(Nb, Nb1, replace=False)]
s_b1 = S1 * np.ones((Nb1, 1))
phi_b1 = ExactPhi(x_b1, L, D, S1, a)
x_s_phi_b1 = np.concatenate((x_b1, s_b1, phi_b1), axis=1)
# 再随机取Nb2个点
x_b2 = (np.setdiff1d(x_b, x_b1).reshape((Nb - Nb1, 1)))[
    np.random.choice(Nb - Nb1, Nb2, replace=False)
]
s_b2 = S2 * np.ones((Nb2, 1))
phi_b2 = ExactPhi(x_b2, L, D, S2, a)
x_s_phi_b2 = np.concatenate((x_b2, s_b2, phi_b2), axis=1)
# 再随机取Nb3个点
x_b3 = (
    np.setdiff1d(x_b, np.concatenate((x_b1, x_b2), axis=0)).reshape((Nb - Nb1 - Nb2, 1))
)[np.random.choice(Nb - Nb1 - Nb2, Nb3, replace=False)]
s_b3 = S3 * np.ones((Nb3, 1))
phi_b3 = ExactPhi(x_b3, L, D, S3, a)
x_s_phi_b3 = np.concatenate((x_b3, s_b3, phi_b3), axis=1)
# 再随机取Nb4个点
x_b4 = (
    np.setdiff1d(x_b, np.concatenate((x_b1, x_b2, x_b3), axis=0)).reshape(
        (Nb - Nb1 - Nb2 - Nb3, 1)
    )
)[np.random.choice(Nb - Nb1 - Nb2 - Nb3, Nb4, replace=False)]
s_b4 = S4 * np.ones((Nb4, 1))
phi_b4 = ExactPhi(x_b4, L, D, S4, a)
x_s_phi_b4 = np.concatenate((x_b4, s_b4, phi_b4), axis=1)
# 剩下的Nb5个点
x_b5 = np.setdiff1d(x_b, np.concatenate((x_b1, x_b2, x_b3, x_b4), axis=0)).reshape(
    (Nb - Nb1 - Nb2 - Nb3 - Nb4, 1)
)
s_b5 = S5 * np.ones((Nb5, 1))
phi_b5 = ExactPhi(x_b5, L, D, S5, a)
x_s_phi_b5 = np.concatenate((x_b5, s_b5, phi_b5), axis=1)

# 将所有的b点合并
x_s_phi_b = np.concatenate(
    (x_s_phi_b1, x_s_phi_b2, x_s_phi_b3, x_s_phi_b4, x_s_phi_b5), axis=0
)

pt_x_s_f = Variable(torch.from_numpy(x_s_f).float(), requires_grad=True).to(device)
pt_x_s_phi_b = Variable(torch.from_numpy(x_s_phi_b).float(), requires_grad=True).to(
    device
)

# 外推边界
z_p_s = np.array(
    [
        [-a / 2 - d, S1],
        [a / 2 + d, S1],
        [-a / 2 - d, S2],
        [a / 2 + d, S2],
        [-a / 2 - d, S3],
        [a / 2 + d, S3],
        [-a / 2 - d, S4],
        [a / 2 + d, S4],
        [-a / 2 - d, S5],
        [a / 2 + d, S5],
    ]
)
pt_z_p_s = Variable(torch.from_numpy(z_p_s).float(), requires_grad=True).to(device)

layers = [2, 100, 100, 100, 100, 100, 100]
head_nums = 4
model = MHPINN(head_nums, layers, L, D, a, d, pt_x_s_f, pt_x_s_phi_b, pt_z_p_s).to(
    device
)
model.apply(custom_weight_init)
max_iters = 100000
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
start_time = time.process_time()

for epoch in range(1, max_iters + 1):
    optimizer.zero_grad()
    loss = model.loss_func()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.losses.append(loss.item())
    model.losses_r.append(model.loss_r.item())
    model.losses_bc1.append(model.loss_bc1.item())
    model.losses_bc2.append(model.loss_bc2.item())
    time_elapsed = time.process_time() - start_time
    if epoch % 500 == 0:
        print(
            "epoch: %d, loss: %.4e, loss_r: %.4e, loss_bc1: %.4e, loss_bc2: %.4e, time: %.4fs"
            % (
                epoch,
                loss.item(),
                model.loss_r.item(),
                model.loss_bc1.item(),
                model.loss_bc2.item(),
                time_elapsed,
            )
        )

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

Ntest = 5000
x = np.linspace(-a / 2, a / 2, Ntest).reshape((Ntest, 1))
s1 = np.ones((Ntest, 1)) * S1
x_s1 = np.concatenate((x, s1), axis=1)
fig = plt.figure()
ax = fig.gca()
Exact = ExactPhi(x, L, D, s1, a)
plt.plot(x, Exact, label="Exact")
plt.savefig("%s/Exact.pdf" % new_folder_path)

fig = plt.figure()
ax = fig.gca()
pt_x_s1 = Variable(torch.from_numpy(x_s1).float(), requires_grad=True).to(device)
u = model.predict(pt_x_s1)
ms_u = u.reshape((Ntest, 1))
plt.plot(x, ms_u, label="MHPINN")
plt.savefig("%s/MHPINN_S1.pdf" % new_folder_path)

# 计算相对L2-error
error = np.sqrt(np.square(ms_u - Exact).sum()) / np.sqrt(np.square(Exact).sum())
print("Error: %.4e" % error)

# 写入结果文件
with open("%s/result.txt" % (new_folder_path), "a") as f:
    f.write("Nf: %d\n" % Nf)
    f.write("Nb: %d\n" % Nb)
    f.write("max_iters: %d\n" % max_iters)
    f.write("Error: %.4e\n" % error)
    f.write("time: %.4fs\n" % time_elapsed)
# plot losses
fig = plt.figure()
ax = fig.gca()
plt.yscale("log")
plt.plot(model.losses, label="loss")
plt.plot(model.losses_r, label="loss_r")
plt.plot(model.losses_bc1, label="loss_bc1")
plt.plot(model.losses_bc2, label="loss_bc2")
plt.legend()
plt.savefig("%s/losses.pdf" % new_folder_path)
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
    "%s/losses_f.txt" % new_folder_path,
    np.array(model.losses_r),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_bc1.txt" % new_folder_path,
    np.array(model.losses_bc1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_bc2.txt" % new_folder_path,
    np.array(model.losses_bc2),
    fmt="%.8e",
    delimiter=",",
)
# 保存参数
np.savetxt(
    "%s/parameters.txt" % new_folder_path,
    np.array([L, D, a, d, layers[-1], head_nums, max_iters, Nf, Nb]),
    fmt="%.8e",
    delimiter=",",
)
# 保存数据
np.savetxt(
    "%s/x_f.txt" % new_folder_path,
    np.array(x_f),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/x_b.txt" % new_folder_path,
    np.array(x_b),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/s.txt" % new_folder_path,
    np.array([S1, S2, S3, S4, S5]),
    fmt="%.8e",
    delimiter=",",
)
