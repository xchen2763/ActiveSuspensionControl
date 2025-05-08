import matplotlib.pyplot as plt
from utils.basics import *
from utils.simulation import *

_, y = discrete_time_simulation(A, B, C, D, u_sb)

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

