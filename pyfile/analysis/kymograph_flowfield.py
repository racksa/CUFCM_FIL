import numpy as np
import matplotlib.pyplot as plt

ur_data = np.load('data/IVP159_flowfield/ur_data_r1.3.npy')
utheta_data = np.load('data/IVP159_flowfield/utheta_data_r1.3.npy')

avg_ur = np.mean(ur_data, axis=0)
avg_utheta = np.mean(utheta_data, axis=0)

t = ur_data.shape[0]/30

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()


ax1.imshow(ur_data.T, cmap='jet', origin='upper', extent=[0, t, 0, 2*np.pi])
ax2.imshow(utheta_data.T, cmap='jet', origin='upper', extent=[0, t, 0, 2*np.pi])

y_ticks = np.linspace(0, 2*np.pi, 5)
y_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ][::-1]
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