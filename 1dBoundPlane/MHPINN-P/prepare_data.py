import numpy as np


def ExactPhi(x, L, D, S, a):
    return (
        S
        * L
        / (2 * D)
        * (np.exp(-np.abs(x) / L) - np.exp(-(a - np.abs(x)) / L))
        / (1 + np.exp(-a / L))
    )


Nf1 = 1000
Nf2 = 1000
Nf3 = 1000
Nf4 = 1000
Nf5 = 1000
Nf = Nf1 + Nf2 + Nf3 + Nf4 + Nf5
Nb1 = 100
Nb2 = 100
Nb3 = 100
Nb4 = 100
Nb5 = 100
Nb = Nb1 + Nb2 + Nb3 + Nb4 + Nb5
L = np.sqrt(0.03)
lambda_tr = 0.0228
D = 1 / 3 * lambda_tr
d = 2 / 3 * lambda_tr
S1 = 1
S2 = 10
S3 = 100
S4 = 1000
S5 = 10000
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
