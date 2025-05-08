import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from utils.basics import *
from utils.simulation import *

n, m = B2.shape
q, _ = C1.shape

X = cp.Variable((n, n), symmetric=True)
W = cp.Variable((q, q), symmetric=True)
Z = cp.Variable((m, n))

objective = cp.Minimize(cp.trace(W))
constraints = [A @ X + B2 @ Z + X @ A.T + Z.T @ B2.T + B1 @ B1.T << -1e-8 * np.eye(n),
               cp.bmat([[W, C1 @ X + D12 @ Z],
                        [(C1 @ X + D12 @ Z).T, X]]) >> 1e-8 * np.eye(q+n)]
prob = cp.Problem(objective, constraints)
result = prob.solve()

print("H2 norm of the closed-loop system = ", np.sqrt(result))
F = Z.value @ np.linalg.inv(X.value)
print("F =", F)

# Discrete-time simulation
A_CL = A + B2 @ F  # closed-loop state matrix
C_CL = C1 + D12 @ F  # closed-loop observability matrix

_, y_sb = discrete_time_simulation(A, B, C, D, u_sb)
_, y_h2_sb = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_sb)

_, y_wave = discrete_time_simulation(A, B, C, D, u_wave)
_, y_h2_wave = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_wave)

_, y_step = discrete_time_simulation(A, B, C, D, u_step)
_, y_h2_step = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_step)

_, y_rand = discrete_time_simulation(A, B, C, D, u_rand)
_, y_h2_rand = discrete_time_simulation(A_CL, B1, C_CL, D11, omega_rand)


eigenvalues, _ = np.linalg.eig(A_CL)
print(eigenvalues)

plt.figure(figsize=(12.8, 9.6))

plt.subplot(4, 2, 1)
plt.plot(time_pts_y, y_sb[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_sb[:, 0], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Speed Bump $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 2)
plt.plot(time_pts_y, y_sb[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_sb[:, 1], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Speed Bump $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 3)
plt.plot(time_pts_y, y_wave[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 0], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Wave Road $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 4)
plt.plot(time_pts_y, y_wave[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 1], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Wave Road $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 5)
plt.plot(time_pts_y, y_step[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_step[:, 0], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Step $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 6)
plt.plot(time_pts_y, y_wave[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_wave[:, 1], label=r'$\mathrm{H_2}$-optimal')
# plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('Step $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 7)
plt.plot(time_pts_y, y_rand[:, 0], label='open-loop')
plt.plot(time_pts_y, y_h2_rand[:, 0], label=r'$\mathrm{H_2}$-optimal')
plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('White Noise $y_1$ ($m$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.subplot(4, 2, 8)
plt.plot(time_pts_y, y_rand[:, 1], label='open-loop')
plt.plot(time_pts_y, y_h2_rand[:, 1], label=r'$\mathrm{H_2}$-optimal')
plt.xlabel('Time ($s$)', fontsize=15)
plt.ylabel('White Noise $y_2$ ($m/s^{2}$)', fontsize=10, labelpad=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(linestyle='--')

plt.show()