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
    def __init__(self, layers, L, D, S, a, d, r_f, r_b, phi_b):
        super(PINN, self).__init__()
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
        loss = 100 * loss_r + 100 * loss_bc1 + loss_bc2
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


Nf = 500
Nb = 50
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
layers = [1, 25, 25, 25, 25, 1]
model = PINN(layers, L, D, S, a, d, pt_x_f, pt_x_b, pt_phi_b).to(device)
model.apply(custom_weight_init)
model.train()

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
plt.plot(x, ms_u, label="PINN")
plt.savefig("%s/Pred.pdf" % new_folder_path)

# 计算相对L2-error
error = np.sqrt(np.square(ms_u - Exact).sum()) / np.sqrt(np.square(Exact).sum())
print("Error: %.4e" % error)

# 写入结果文件
with open("%s/result.txt" % (new_folder_path), "a") as f:
    f.write("Nf: %d\n" % Nf)
    f.write("Nb: %d\n" % Nb)
    f.write("max_iters: %d\n" % model.max_iter_adam)
    f.write("Error: %.4e\n" % error)
    f.write("time: %.4fs\n" % (model.end_time - model.init_time))
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
    np.array(model.losses_f),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_b1.txt" % new_folder_path,
    np.array(model.losses_b1),
    fmt="%.8e",
    delimiter=",",
)
np.savetxt(
    "%s/losses_b2.txt" % new_folder_path,
    np.array(model.losses_b2),
    fmt="%.8e",
    delimiter=",",
)
# 保存参数
np.savetxt(
    "%s/parameters.txt" % new_folder_path,
    np.array([L, D, S, a, d, layers[-2]]),
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
