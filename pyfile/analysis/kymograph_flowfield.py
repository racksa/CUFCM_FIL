import numpy as np
import matplotlib.pyplot as plt


n = 159

ur_data = np.load(f'data/IVP159_flowfield/ur_data_fil{n}_r1.3.npy')
utheta_data = np.load(f'data/IVP159_flowfield/utheta_data_fil{n}_r1.3.npy')
grid_shape = np.load(f'data/IVP159_flowfield/grid_shape_fil{n}_r1.3.npy')

n_frame = ur_data.shape[0]
n_r, n_phi, n_theta = grid_shape
print(n_r, n_phi, n_theta)

# assuming n_r = 1
reshaped_ur_data = ur_data.reshape(n_frame, n_theta, n_phi)
avg_ur_data = reshaped_ur_data.mean(axis=1)

reshaped_utheta_data = utheta_data.reshape(n_frame, n_theta, n_phi)
avg_utheta_data = reshaped_utheta_data.mean(axis=1)
print(avg_ur_data)

t = n_frame/30

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()


ax1.imshow(avg_ur_data.T, cmap='jet', origin='upper', extent=[0, t, 0, 2*np.pi])
ax2.imshow(avg_utheta_data.T, cmap='jet', origin='upper', extent=[0, t, 0, 2*np.pi])

y_ticks = np.linspace(0, 2*np.pi, 5)
y_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ][::-1]
y_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$' ][::-1]
ax1.set_yticks(ticks=y_ticks, labels=y_labels)
ax2.set_yticks(ticks=y_ticks, labels=y_labels)

ax1.set_xlabel(r'$t/T$')
ax1.set_ylabel(r'$\theta$')

ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'$\theta$')

plt.tight_layout()
fig1.savefig(f'fig/ur.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/utheta.pdf', bbox_inches = 'tight', format='pdf')

plt.show()