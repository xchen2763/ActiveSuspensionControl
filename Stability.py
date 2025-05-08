import cvxpy as cp
import numpy as np
from utils.basics import *

n = A.shape[0]
# Define a symmetric matrix variable P
P = cp.Variable((n, n), symmetric=True)

# Define the Lyapunov matrix: L = A^T * P + P * A
L = A.T @ P + P @ A

# Set up the constraints: P >> 0 and L << 0.
constraints = [P >> 1e-6 * np.eye(n),  # Ensure P is positive definite
               L == - np.eye(n)]  # Ensure negative definiteness

# Formulate the problem. We don't have an objective (feasibility problem).
prob = cp.Problem(cp.Minimize(0), constraints)

# Solve the problem
result = prob.solve()

np.set_printoptions(precision=2, suppress=True)

if prob.status in ["optimal", "optimal_inaccurate"]:
    print("A valid P was found, so the system is stable.")
    print("P =", P.value)
    print("Q =", -L.value)
else:
    print("No valid P was found; the Lyapunov condition might not hold.")