import numpy as np

dt = 0.00025  # simulation time step

time_steps = 20000   # number of time steps, 5 seconds in total
time_pts_x = np.linspace(0, dt * time_steps, time_steps + 1)  # mark time points for plotting
time_pts_y = np.arange(0, dt * time_steps, dt)

# Profile of sinusoidal speed bump
t_sb = 0.25
omega_sb = np.zeros((time_steps, 1))
for i in range(int(t_sb / dt)):
    omega_sb[i, 0] = 0.025 * (1 - np.cos(8 * np.pi * i * dt))
u_sb = np.hstack((omega_sb, np.zeros((time_steps, 1))))

# Profile of continuous wave
omega_wave = np.zeros((time_steps, 1))
for i in range(len(omega_wave)):
    omega_wave[i, 0] = 0.025 * (1 - np.cos(8 * np.pi * i * dt))
u_wave = np.hstack((omega_wave, np.zeros((time_steps, 1))))

# Profile of step
t_step = 1
omega_step = 0.05 * np.ones((time_steps, 1))
# omega_step[int(t_step / dt): , 0] = 0.05
u_step = np.hstack((omega_step, np.zeros((time_steps, 1))))

# Random white noise road profile
omega_rand = 0.05 * np.random.randn(time_steps).reshape(-1, 1)
u_rand = np.hstack((omega_rand, np.zeros((time_steps, 1))))


def discrete_time_simulation(A, B, C, D, u):
    n, m = B.shape
    p, _ = C.shape
    x = np.zeros((time_steps + 1, n))
    y = np.zeros((time_steps, p))

    for i in range(time_steps):
        dx = A @ x[i] + B @ u[i]  # shape of u should be (time_steps, m)
        x[i+1] = x[i] + dx * dt
        y[i] = C @ x[i] + D @ u[i]
    
    return x, y


def h_inf_output_feedback_simulation(A, B1, B2, C1, D11, D12, AK, BK, CK, DK, omega):
    n, m = B2.shape  # n = 4, m = 1
    p, _ = C1.shape  # p = 2

    x = np.zeros((time_steps + 1, n))
    xk = np.zeros((time_steps + 1, n))
    u = np.zeros((time_steps + 1, m))
    y = np.zeros((time_steps, p))

    for i in range(time_steps):
        # Plant G
        dx = A @ x[i] + B1 @ omega[i] + B2 @ u[i]
        x[i+1] = x[i] + dx * dt
        y[i] = C1 @ x[i] + D11 @ omega[i] + D12 @ u[i]

        # Controller K
        dxk = AK @ xk[i] + BK @ y[i]
        xk[i+1] = xk[i] + dxk * dt
        u[i+1] = CK @ xk[i] + DK @ y[i]
    
    return x, y


# SVD matrix decomposition M = AB
def svd_decompose(M):
    U, s, Vt = np.linalg.svd(M)
    S_sqrt = np.diag(np.sqrt(s))
    A = U @ S_sqrt
    B = S_sqrt @ Vt
    return A, B