import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils.basics import *

np.set_printoptions(precision=2, suppress=True)

C_CA = np.concatenate((C, C@A, C@A@A, C@A@A@A), axis=0)
print("Observability Matrix C_CA =", C_CA)

rank = np.linalg.matrix_rank(C_CA)
print(rank)

# build a Luenberger observer
n, m = B.shape
p, _ = C.shape
W = cp.Variable((n, n), symmetric=True)
V = cp.Variable((p, n))

constraints = [W >> 1e-6 * np.eye(n),
               A.T @ W + W @ A + C.T @ V + V.T @ C << -1e-6 * np.eye(n)]

prob = cp.Problem(cp.Minimize(0), constraints)
result = prob.solve()

L = np.linalg.inv(W.value) @ V.value.T
print('L =', L)
A_LC = A + L @ C
B_LD = B + L @ D

# discrete-time simulation
dt = 0.00025  # simulation time step
t_speedbump = 0.25  # duration of the quarter-car system driving over the speed bump
time_steps = 20000   # number of time steps, 5 seconds in total
time_pts = np.linspace(0, dt * time_steps, time_steps + 1)  # mark time points for plotting

x = np.zeros((time_steps + 1, n))  # actual states
x[0] = [0.1, 0.1, 0.1, 0.1]  # assume initial offset
x_hat = np.zeros((time_steps + 1, n))  # observed states 
y = np.zeros((time_steps, p))  # output
u = np.zeros((time_steps, m))  # open-loop simulation, input only contains road height change for now
for i in range(int(t_speedbump / dt)):
    u[i, 0] = 0.025 * (1 - np.cos(8 * np.pi * i * dt))

for i in range(time_steps):
    dx = A @ x[i] + B @ u[i]
    x[i+1] = x[i] + dx * dt

    y[i] = C @ x[i] + D @ u[i]

    dx_hat = A_LC @ x_hat[i] - L @ y[i] + B_LD @ u[i]
    x_hat[i+1] = x_hat[i] + dx_hat * dt

plt.figure()

plt.subplot(4, 1, 1)
plt.plot(time_pts, x[:, 0], label=r'$x_1$')
plt.plot(time_pts, x_hat[:, 0], label=r'$\hat{x}_1$')
plt.ylabel(r'$x_1$', fontsize=20)
plt.tick_params(labelsize=15)
# plt.title('Open-loop System Simulation', fontsize=30)
plt.legend(fontsize=15, loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time_pts, x[:, 1], label=r'$x_2$')
plt.plot(time_pts, x_hat[:, 1], label=r'$\hat{x}_2$')
plt.ylabel(r'$x_2$', fontsize=20)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time_pts, x[:, 2], label=r'$x_3$')
plt.plot(time_pts, x_hat[:, 2], label=r'$\hat{x}_3$')
plt.ylabel(r'$x_3$', fontsize=20)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time_pts, x[:, 3], label=r'$x_4$')
plt.plot(time_pts, x_hat[:, 3], label=r'$\hat{x}_4$')
plt.ylabel(r'$x_4$', fontsize=20)
plt.xlabel('time', fontsize=20)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.grid(True)

plt.show()
