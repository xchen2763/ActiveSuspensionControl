import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from utils.basics import *
from utils.simulation import *

np.set_printoptions(precision=1, suppress=True)

# 1. Solve LMIs for X1 and Y1
gamma = cp.Variable()
An = cp.Variable((4, 4))
Bn = cp.Variable((4, 2))
Cn = cp.Variable((1, 4))
Dn = cp.Variable((1, 2))
X1 = cp.Variable((4, 4), symmetric=True)
Y1 = cp.Variable((4, 4), symmetric=True)

M11 = A @ Y1 + Y1 @ A.T + B2 @ Cn + Cn.T @ B2.T
M21 = A.T + An + (B2 @ Dn @ C2).T
M22 = X1 @ A + A.T @ X1 + Bn @ C2 + C2.T @ Bn.T
M31 = (B1 + B2 @ Dn @ D21).T
M32 = (X1 @ B1 + Bn @ D21).T
M41 = C1 @ Y1 + D12 @ Cn
M42 = C1 + D12 @ Dn @ C2
M43 = D11 + D12 @ Dn @ D21

M = cp.bmat([[M11, M21.T, M31.T, M41.T],
             [M21, M22, M32.T, M42.T],
             [M31, M32, -gamma*np.eye(1), M43.T],
             [M41, M42, M43, -gamma*np.eye(2)]])

N = cp.bmat([[Y1, np.eye(4)],
             [np.eye(4), X1]])

constraints = [M << -1e-2 * np.eye(11),
               N >> 1e-2 * np.eye(8)]

prob = cp.Problem(cp.Minimize(gamma), constraints)
prob.solve()

# eigenvalues, _ = np.linalg.eig(M.value)
# print(eigenvalues)

print(gamma.value)
print(np.eye(4) - X1.value @ Y1.value)

# 2. Choose X2 and Y2
# Y2 = np.eye(4)
# X2 = np.eye(4) - X1.value @ Y1.value

X2, Y2 = svd_decompose(np.eye(4) - X1.value @ Y1.value)
Y2 = Y2.T

# 3. Solve AK2, BK2, CK2, DK2
M1 = np.block([[X2, X1.value @ B2],
               [np.zeros((1, 4)), np.eye(1)]])

Mn = np.block([[An.value, Bn.value],
               [Cn.value, Dn.value]])

M2 = np.block([[X1.value @ A @ Y1.value, np.zeros((4, 2))],
               [np.zeros((1, 4)), np.zeros((1, 2))]])

M3 = np.block([[Y2.T, np.zeros((4, 2))],
               [C2 @ Y1.value, np.eye(2)]])

MK2 = np.linalg.inv(M1) @ (Mn - M2) @ np.linalg.inv(M3)

AK2 = MK2[:4, :4]
BK2 = MK2[:4, 4:]
CK2 = MK2[4:, :4]
DK2 = MK2[4:, 4:]

# 4. Solve AK, BK, CK, DK
DK = np.linalg.inv(np.eye(1) + DK2 @ D22) @ DK2
BK = BK2 @ (np.eye(2) - D22 @ DK)
CK = (np.eye(1) - DK @ D22) @ CK2
AK = AK2 - BK @ np.linalg.inv(np.eye(2) - D22 @ DK) @ D22 @ CK

print('AK =', AK)
print('BK =', BK)
print('CK =', CK)
print('DK =', DK)
eigenvalues, _ = np.linalg.eig(AK)
print(eigenvalues)

# 5. Discrete-time simulation
_, y = h_inf_output_feedback_simulation(A, B1, B2, C1, D11, D12, AK, BK, CK, DK, omega_sb)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(time_pts_y, y[:, 0])
plt.ylabel('Body Movement ($m$)', fontsize=20, labelpad=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=12)
plt.grid(linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(time_pts_y, y[:, 1])
plt.xlabel('Time ($s$)', fontsize=20)
plt.ylabel('Body Acceleration ($m/s^{2}$)', fontsize=20, labelpad=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=12)
plt.grid(linestyle='--')

plt.show()