"""
Estimate Gaussian beam radius using knife-edge method
with background correction
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from uncertainties import ufloat

# -----------------------------
# Experimental Data
# -----------------------------

x = np.array([
    0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,
    4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5
])  # Position (mm)

P = np.array([
    2.05, 2.035, 2.05, 2.047, 2.011, 2.025, 1.955, 1.86, 1.575,
    1.14, 0.705, 0.316, 0.085, 0.016, 0.008, 0.004, 0.002, 0.002
])  # Power (W)

dP = 2*np.array([
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.010, 0.005, 0.010,
    0.005, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001
])  # Power uncertainty (W)

# -----------------------------
# Knife-edge Model with Background
# -----------------------------
def knife_edge(x, P0, Pmax, x0, w0):
    """
    P0   : background power (W)
    Pmax : maximum beam power contribution (W)
    x0   : beam center position (mm)
    w0   : 1/e^2 beam waist (mm)
    """
    return P0 + 0.5 * Pmax * (1 - erf(np.sqrt(2) * (x - x0) / w0))

# -----------------------------
# Curve Fitting
# -----------------------------
initial_guess = [0.0, max(P), 4.5, 1.0]  # [P0, Pmax, x0, w0]

popt, pcov = curve_fit(
    knife_edge,
    x,
    P,
    sigma=dP,
    p0=initial_guess,
    absolute_sigma=True
)

P0_fit, Pmax_fit, x0_fit, w0_fit = popt
perr = np.sqrt(np.diag(pcov))

# -----------------------------
# Beam Waist with Uncertainty
# -----------------------------
w0_fit_u = ufloat(w0_fit, perr[3])
resolution = 0.01  # mm, micrometer resolution
w0 = w0_fit_u 

# -----------------------------
# Print Results
# -----------------------------
print(f"Background P0 = {P0_fit:.4f} ± {perr[0]:.4f} W")
print(f"Maximum beam power Pmax = {Pmax_fit:.4f} ± {perr[1]:.4f} W")
print(f"Beam center x0 = {x0_fit:.4f} ± {perr[2]:.4f} mm")
print(f"Beam waist w0 = {w0:.1uS} mm")

# -----------------------------
# Plot Data and Fit
# -----------------------------
x_fit = np.linspace(min(x), max(x), 400)
P_fit = knife_edge(x_fit, *popt)

plt.figure(figsize=(7, 5))
plt.errorbar(x, P, yerr=dP, fmt='o', capsize=3, label='Data')
plt.plot(x_fit, P_fit, 'r-', label='Fit')
plt.xlabel('Position (mm)')
plt.ylabel('Power (W)')
plt.title('Knife-edge Measurement of Beam Waist with Background')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()