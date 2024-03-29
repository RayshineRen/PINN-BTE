import torch
import torch.nn as nn
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
        self.input_layer = nn.Linear(layers[0], layers[1])
        self.activation = nn.Tanh()
        for i in range(1, len(layers) - 2):
            setattr(self, "hidden_layer" + str(i), nn.Linear(layers[i], layers[i + 1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, r):
        if len(self.layers) > 2:
            x = self.activation(self.input_layer(r))
            for i in range(1, len(self.layers) - 2):
                x = self.activation(getattr(self, "hidden_layer" + str(i))(x))
            y = self.output_layer(x)
        if len(self.layers) == 2:
            y = self.output_layer(r)
        return y


class MHPINN(nn.Module):
    def __init__(self, head_nums, layers, L, D, S, a, d, r_f, r_b, phi_b):
        super(MHPINN, self).__init__()
        self.head_nums = head_nums
        self.neural_nets = nn.ModuleList(
            [NeuralNet(layers).to(device) for _ in range(self.head_nums)]
        )
        self.end_net = NeuralNet([layers[-1] * head_nums, 1]).to(device)
        self.max_iter_adam = 100000
        self.L = L
        self.D = D
        self.S = S
        self.a = a
        self.d = d
        self.r_f = r_f
        self.r_b = r_b
        self.phi_b = phi_b
        self.z_p1 = Variable(
            torch.from_numpy(np.array([[-a / 2 - d]])).float(), requires_grad=True
        ).to(device)
        self.z_p2 = Variable(
            torch.from_numpy(np.array([[a / 2 + d]])).float(), requires_grad=True
        ).to(device)
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
        return -self.D * phi_r - self.S / 2.0

    def loss_func(self):
        f_pred = self.f(self.r_f)
        # bc_pred = self.bc(self.r_b)
        bc_pred = self.forward(self.r_b)
        self.loss_r = torch.mean(torch.square(f_pred))
        self.loss_bc1 = torch.mean(torch.square(bc_pred - self.phi_b))
        self.loss_bc2 = torch.mean(torch.square(self.forward(self.z_p2))) + torch.mean(
            torch.square(self.forward(self.z_p1))
        )
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


Nf = 5000
Nb = 500
L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
S = 1
a = 2.0
b_l_minus = -0.1
b_r_minus = -0.01
b_l_posit = +0.01
b_r_posit = +0.1
x_f_l = np.random.rand(Nf // 2) * (-a / 2 - b_l_minus) + b_l_minus
x_f_r = np.random.rand(Nf // 2) * (a / 2 - b_r_posit) + b_r_posit
x_f = np.concatenate((x_f_l, x_f_r), axis=0).reshape((Nf, 1))
x_b_l = np.random.rand(Nb // 2) * (b_l_minus - b_r_minus) + b_r_minus
x_b_r = np.random.rand(Nb // 2) * (b_r_posit - b_l_posit) + b_l_posit
x_b = np.concatenate((x_b_l, x_b_r), axis=0).reshape((Nb, 1))
phi_b = ExactPhi(x_b, L, D, S, a)
pt_x_f = Variable(torch.from_numpy(x_f).float(), requires_grad=True).to(device)
pt_x_b = Variable(torch.from_numpy(x_b).float(), requires_grad=True).to(device)
pt_phi_b = Variable(torch.from_numpy(phi_b).float(), requires_grad=True).to(device)
layers = [1, 25, 25, 25, 25]
head_nums = 4
model = MHPINN(head_nums, layers, L, D, S, a, d, pt_x_f, pt_x_b, pt_phi_b).to(device)
model.apply(custom_weight_init)
max_iters = 100000
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
fig = plt.figure()
ax = fig.gca()
Exact = ExactPhi(x, L, D, S, a)
plt.plot(x, Exact, label="Exact")
plt.savefig("%s/Exact.pdf" % new_folder_path)

fig = plt.figure()
ax = fig.gca()
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
u = model.predict(pt_x)
ms_u = u.reshape(pt_x.shape)
plt.plot(x, ms_u, label="WPINN")
plt.savefig("%s/MHPred.pdf" % new_folder_path)

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
    np.array([L, D, S, a, d, Nf, Nb, head_nums, layers[-1]]),
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
