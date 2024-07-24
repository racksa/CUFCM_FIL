import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 24})

omega0 = 2*np.pi
viscosity = 1
L = 2.6*(20-1)

phase_forcing_filename = 'input/forcing/fulford_and_blake_reference_phase_generalised_forces_NSEG=20_SEP=2.600000.dat'
angle_forcing_filename = 'input/forcing/fulford_and_blake_reference_angle_generalised_forces_NSEG=20_SEP=2.600000.dat'

phase_forcing_f = open(phase_forcing_filename)
angle_forcing_f = open(angle_forcing_filename)

phase_forcing_str = phase_forcing_f.readline()
angle_forcing_str = angle_forcing_f.readline()

n_phase = int(phase_forcing_str.split()[0])

phase_forcing = np.array(phase_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
angle_forcing = np.array(angle_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
phase_mean = np.mean(phase_forcing)
angle_mean = np.mean(angle_forcing)

print(phase_mean)

t = np.linspace(0, 1, n_phase+1)[:-1]
signal_fft = np.fft.fft(phase_forcing-phase_mean)
frequencies = np.fft.fftfreq(n_phase, t[1] - t[0])
# Extract the Fourier coefficients (A_n and B_n)
A_n = 2 * np.real(signal_fft[:n_phase//2]) / n_phase
B_n = -2 * np.imag(signal_fft[:n_phase//2]) / n_phase
freqs = frequencies[:n_phase//2]

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()

# Reconstruct the signal using the Fourier coefficients
def signal_component(An, Bn, freq, t, n):
    return An[n] * np.cos(2 * np.pi * freq[n] * t) + Bn[n] * np.sin(2 * np.pi * freq[n] * t)

reconstructed_signal = np.zeros(n_phase)
truncate = 12
for n in range(1, truncate):
    ax2.scatter(np.linspace(0, 2*np.pi, n_phase+1)[:-1], signal_component(A_n, B_n, freqs, t, n))
    reconstructed_signal += signal_component(A_n, B_n, freqs, t, n)




ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], phase_forcing, label=r'$Q_\psi$')
ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], angle_forcing, label=r'$Q_\theta$')
ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], reconstructed_signal+phase_mean, label=r'$Q_\theta$')

ax1.set_xlabel(r'$\psi$')
ax1.set_ylabel(r'$\frac{Q}{\omega \eta L^3}$')

ax1_x_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ]
ax1.set_xticks(ticks= np.linspace(0, 2*np.pi, 5), labels=ax1_x_labels)


# ax2.plot(A_n[:8])
# ax2.plot(B_n[:8])
# ax2.scatter(frequencies, np.abs(signal_fft))

ax1.set_xlim(0, 2*np.pi)
ax1.legend()
plt.tight_layout()
fig1.savefig(f'fig/forcing.pdf', bbox_inches = 'tight', format='pdf')
plt.show()


#