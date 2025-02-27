import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

# Path to the directory where fonts are stored
font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
# Choose the TTF or OTF version of CMU Serif Regular
font_path = os.path.join(font_dir, 'cmunrm.ttf')  # Or 'cmunrm.otf' if you prefer OTF
# Load the font into Matplotlib's font manager
prop = fm.FontProperties(fname=font_path)
# Register each font file with Matplotlib's font manager
for font_file in os.listdir(font_dir):
    if font_file.endswith('.otf'):
        fm.fontManager.addfont(os.path.join(font_dir, font_file))
# Set the global font family to 'serif' and specify CMU Serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Use 'cm' for Computer Modern
plt.rcParams.update({'font.size': 24})


omega0 = 2*np.pi
viscosity = 1
L = 2.6*(20-1)

# phase_forcing_filename = 'input/forcing/fulford_and_blake_reference_phase_generalised_forces_NSEG=20_SEP=2.600000.dat'
# angle_forcing_filename = 'input/forcing/fulford_and_blake_reference_angle_generalised_forces_NSEG=20_SEP=2.600000.dat'

phase_forcing_filename = 'input/forcing/fulford_and_blake_original_reference_phase_generalised_forces_NSEG=20_SEP=2.600000.dat'
angle_forcing_filename = 'input/forcing/fulford_and_blake_original_reference_angle_generalised_forces_NSEG=20_SEP=2.600000.dat'
real_force_filename = 'data/regular_wall_sim/20250204_1e-4_ref/ciliate_1fil_0blob_1.00R_0.0500torsion_0.0000tilt_1.0000dp_0.0000noise_0.0000ospread_seg_forces.dat'

# phase_forcing_filename = 'input/forcing/volvox_reference_phase_generalised_forces_NSEG=20_SEP=2.600000.dat'
# angle_forcing_filename = 'input/forcing/volvox_reference_angle_generalised_forces_NSEG=20_SEP=2.600000.dat'


phase_forcing_f = open(phase_forcing_filename)
angle_forcing_f = open(angle_forcing_filename)
real_force_f = open(real_force_filename)

phase_forcing_str = phase_forcing_f.readline()
angle_forcing_str = angle_forcing_f.readline()
real_force_str = real_force_f.readlines()

n_phase = int(phase_forcing_str.split()[0])

phase_forcing = np.array(phase_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
angle_forcing = np.array(angle_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
real_force = np.array([s.split()[1:] for s in real_force_str], dtype=float)


phase_mean = np.mean(phase_forcing)
angle_mean = np.mean(angle_forcing)
real_total_force = np.zeros(n_phase)
for t in range(n_phase):
    for i in range(20):
        real_total_force[t] += np.mean(np.linalg.norm(real_force[t][6*i: 6*i+3]))
real_total_force_mean = np.mean(real_total_force)

phase_forcing -= - phase_mean

# fourier transform of the generalised force
t = np.linspace(0, 1, n_phase+1)[:-1]
signal_fft_phase = np.fft.fft(phase_forcing)
signal_fft_angle = np.fft.fft(angle_forcing)
frequencies = np.fft.fftfreq(n_phase, t[1] - t[0])
# Extract the Fourier coefficients (A_n and B_n)
A_n_phase = 2 * np.real(signal_fft_phase[:n_phase//2]) / n_phase
B_n_phase = -2 * np.imag(signal_fft_phase[:n_phase//2]) / n_phase
A_n_angle = 2 * np.real(signal_fft_angle[:n_phase//2]) / n_phase
B_n_angle = -2 * np.imag(signal_fft_angle[:n_phase//2]) / n_phase

freqs = frequencies[:n_phase//2]

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig3 = plt.figure()
ax3 = fig3.add_subplot()
fig4 = plt.figure()
ax4 = fig4.add_subplot()

# Reconstruct the signal using the Fourier coefficients
def signal_component(An, Bn, freq, t, n):
    return An[n] * np.cos(2 * np.pi * freq[n] * t) + Bn[n] * np.sin(2 * np.pi * freq[n] * t)

reconstructed_signal = np.zeros(n_phase)
truncate = 12
for n in range(1, truncate):
    ax2.scatter(np.linspace(0, 2*np.pi, n_phase+1)[:-1], signal_component(A_n_phase, B_n_phase, freqs, t, n))
    reconstructed_signal += signal_component(A_n_phase, B_n_phase, freqs, t, n)




ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], phase_forcing, label=r'$Q_1$', c='black', linestyle='solid')
ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], angle_forcing, label=r'$Q_2$', c='black', linestyle='dashed')
# ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], reconstructed_signal+phase_mean, label=r'$Q_\theta$')

# Add comparison with another forcing data file.
# phase_forcing_filename = 'input/forcing/fulford_and_blake_original_reference_phase_generalised_forces_NSEG=20_SEP=2.600000.dat'
# angle_forcing_filename = 'input/forcing/fulford_and_blake_original_reference_angle_generalised_forces_NSEG=20_SEP=2.600000.dat'
# phase_forcing_f = open(phase_forcing_filename)
# angle_forcing_f = open(angle_forcing_filename)
# phase_forcing_str = phase_forcing_f.readline()
# angle_forcing_str = angle_forcing_f.readline()
# n_phase = int(phase_forcing_str.split()[0])
# phase_forcing = np.array(phase_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
# angle_forcing = np.array(angle_forcing_str.split()[1:], dtype=float)/omega0/viscosity/L**3
# phase_mean = np.mean(phase_forcing)
# angle_mean = np.mean(angle_forcing)
# ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], phase_forcing, label=r'$Q_\psi$')
# ax1.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], angle_forcing, label=r'$Q_\theta$')

ax1.set_xlabel(r'$\psi_1$')
ax1.set_ylabel(r'$Q/\omega \eta L^3$')
ax1_x_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ]
ax1.set_xticks(ticks= np.linspace(0, 2*np.pi, 5), labels=ax1_x_labels)
ax1.set_xlim(0, 2*np.pi)
ax1.legend(frameon=False)

# ax2.plot(A_n[:8])
# ax2.plot(B_n[:8])
# ax2.scatter(frequencies, np.abs(signal_fft))


ax3.plot(np.linspace(0, 2*np.pi, n_phase+1)[:-1], real_total_force)
ax3.set_xlabel(r'$\psi_1$')
ax3.set_ylabel(r'$\sum_n |\lambda_n|$')
ax3_x_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ]
ax3.set_xticks(ticks= np.linspace(0, 2*np.pi, 5), labels=ax1_x_labels)
ax3.set_xlim(0, 2*np.pi)

# real_total_force -= real_total_force_mean

# fourier transform of the real force
t = np.linspace(0, 1, n_phase+1)[:-1]
signal_fft = np.fft.fft(real_total_force)
frequencies = np.fft.fftfreq(n_phase, t[1] - t[0])
# Extract the Fourier coefficients (A_n and B_n)
A_n_force = 2 * np.real(signal_fft[:n_phase//2]) / n_phase
B_n_force  = -2 * np.imag(signal_fft[:n_phase//2]) / n_phase
freqs = frequencies[:n_phase//2]

coeff_lim = 8
ax4.plot(np.sqrt(A_n_phase[:coeff_lim]**2 + B_n_phase[:coeff_lim]**2), label=r'$Q_1$', c='black', linestyle='solid', marker='+')
ax4.plot(np.sqrt(A_n_angle[:coeff_lim]**2 + B_n_angle[:coeff_lim]**2), label=r'$Q_2$', c='black', linestyle='dashed', marker='+')
ax4.set_xlabel(r'$n$')
ax4.set_ylabel(r'$Coefficient$')
ax4.set_xlim(0)
ax4.set_ylim(0)
ax4.set_xticks(ticks= np.linspace(0, coeff_lim-1, coeff_lim))
ax4.legend(frameon=False)

fig1.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig1.savefig(f'fig/forcing.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/real_force.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/forcing_fourier_coeffs.pdf', bbox_inches = 'tight', format='pdf')
plt.show()


#