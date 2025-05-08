import cvxpy as cp
import numpy as np
from utils.basics import *

np.set_printoptions(precision=3, suppress=True)

n = A.shape[0]
# Define a symmetric matrix variable P
X = cp.Variable((n, n), symmetric=True)

# Set up the constraints: P >> 0 and L << 0.
constraints = [A @ X + X @ A.T + B @ B.T == 0,
               X >> 1e-6 * np.eye(n)]

# Formulate the problem. We don't have an objective (feasibility problem).
prob = cp.Problem(cp.Minimize(0), constraints)

# Solve the problem
result = prob.solve()

if prob.status in ["optimal", "optimal_inaccurate"]:
    print("Positive definite X was found, so the system is controllable.")
    print("X =", X.value)
    eigenvalues, _ = np.linalg.eig(X.value)
    print("Eigenvalues of X:", eigenvalues)
else:
    print("No valid X was found; the Lyapunov condition might not hold.")