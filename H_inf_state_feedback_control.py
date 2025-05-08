import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from utils.basics import *
from utils.simulation import *

D11 = np.zeros((2, 1))

Y = cp.Variable((4, 4), symmetric=True)
Z = cp.Variable((1, 4))
gamma = cp.Variable()

M = cp.bmat([[Y @ A.T + A @ Y + Z.T @ B2.T + B2 @ Z, B1, Y @ C1.T + Z.T @ D12.T],
             [B1.T, - gamma * np.eye(1), D11.T],
             [C1 @ Y + D12 @ Z, D11, -gamma * np.eye(2)]])

constraints = [Y >> 1e-5 * np.eye(4),
               M << -1e-5 * np.eye(7)]

prob = cp.Problem(cp.Minimize(gamma), constraints)
prob.solve()

print('Optimum closed-loop H_inf norm is: {:.3f}'.format(gamma.value))
F = Z.value @ np.linalg.inv(Y.value)
print("F =", F)

A_CL = A + B2 @ F
C_CL = C1 + D12 @ F

_, y_sb = discrete_time_simulation(A, B, C, D, u_sb)
_, y_h2_sb = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_sb)

_, y_wave = discrete_time_simulation(A, B, C, D, u_wave)
_, y_h2_wave = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_wave)

_, y_step = discrete_time_simulation(A, B, C, D, u_step)
_, y_h2_step = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_step)

_, y_rand = discrete_time_simulation(A, B, C, D, u_rand)
_, y_h2_rand = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_rand)

# Visualization
plt.figure(figsize=(12.8, 9.6))

plt.subplot(4, 2, 1)
plt.plot(time_pts_y, y_sb[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_sb[:, 0], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Speed Bump $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 2)
plt.plot(time_pts_y, y_sb[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_sb[:, 1], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Speed Bump $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 3)
plt.plot(time_pts_y, y_wave[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 0], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Wave Road $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 4)
plt.plot(time_pts_y, y_wave[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 1], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Wave Road $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 5)
plt.plot(time_pts_y, y_step[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_step[:, 0], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Step $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 6)
plt.plot(time_pts_y, y_wave[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 1], label=r'$\mathrm{H_\infty}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Step $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 7)
plt.plot(time_pts_y, y_rand[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_rand[:, 0], label=r'$\mathrm{H_\infty}$-optimal')
plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('White Noise $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 8)
plt.plot(time_pts_y, y_rand[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_rand[:, 1], label=r'$\mathrm{H_\infty}$-optimal')
plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('White Noise $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.show()