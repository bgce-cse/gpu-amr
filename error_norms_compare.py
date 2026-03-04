import numpy as np

# -----------------------------
# Load data (written by C++ runs)
# -----------------------------

rho10 = np.loadtxt("/root/gpu-amr/build/rho_N10.txt").reshape(10, 10)
rho20 = np.loadtxt("/root/gpu-amr/build/rho_N20.txt").reshape(20, 20)
rho40 = np.loadtxt("/root/gpu-amr/build/rho_N40.txt").reshape(40, 40)

# -----------------------------
# Proper restriction (2x2 average)
# -----------------------------
def restrict2(a):
    return 0.25 * (
        a[0::2, 0::2] +
        a[1::2, 0::2] +
        a[0::2, 1::2] +
        a[1::2, 1::2]
    )

rho20_to_10 = restrict2(rho20)
rho40_to_20 = restrict2(rho40)
rho40_to_10 = restrict2(rho40_to_20)

# -----------------------------
# RMS (normalized L2) error
# -----------------------------
def rms_error(a, b):
    return np.sqrt(np.mean((a - b)**2))

E10 = rms_error(rho10, rho20_to_10)
E20 = rms_error(rho20, rho40_to_20)
E40 = rms_error(rho10, rho40_to_10)

rate_10_20 = np.log2(E10 / E20)
rate_20_40 = np.log2(E20 / E40)

# -----------------------------
# Output
# -----------------------------
print("E10 =", E10)
print("E20 =", E20)
print("E40 =", E40)
print("rate(10→20) =", rate_10_20)
print("rate(20→40) =", rate_20_40)

