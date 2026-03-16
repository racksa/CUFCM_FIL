import numpy as np

filename = 'pos8.fcm'
data = np.genfromtxt(filename, delimiter=' ', dtype=float)
V_UAMMD = data[:,-7:-4]
W_UAMMD = data[:, -3:]

N=len(data)


filename = 'simulation_data_N500000_ngd19_sigmafac1.00_Rfac2.8_NPTS588_Nproc32'
data = np.genfromtxt(filename, delimiter=' ', dtype=float, skip_header=6)
V_FCM = data[:, -6:-3]
W_FCM = data[:, -3:]


def modulus(array1):
    array2 = array1*array1
    array2_sum = np.sum(array2, axis=1)
    array_modulus = np.sqrt(array2_sum)
    return array_modulus

def diff_mod(array1, array2):
    diff = array1 - array2
    # diff2 = diff*diff
    # diff2_sum = np.sum(diff2, axis=1)
    # diff_modulus = np.sqrt(diff2_sum)
    return modulus(diff)

def relative_error(array1, array2):
    modulus_fraction = diff_mod(array1, array2) / modulus(array1)
    return modulus_fraction

# print(np.sum(relative_error(V_UAMMD, V_FCM))/N)
print(np.sum(relative_error(W_UAMMD, W_FCM))/N)
