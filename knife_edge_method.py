import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from uncertainties import ufloat

# -----------------------------
# Experimental Data
# -----------------------------

# Position (mm) — measured using micrometer (resolution ≈ 0.01 mm)
x = np.array([
    0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,
    4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5
])

# Measured optical power (W)
P = np.array([
    2.05, 2.035, 2.05, 2.047, 2.011, 2.025, 1.955, 1.86, 1.575,
    1.14, 0.705, 0.316, 0.085, 0.016, 0.008, 0.004, 0.002, 0.002
])

# Uncertainty in power measurement (W)
dP = np.array([
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.001, 0.005, 0.001,
    0.005, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001
])

# -----------------------------
# Knife-edge Model
# -----------------------------
# Based on Gaussian beam with 1/e^2 waist definition
def knife_edge(x, P0, x0, w0):
    return 0.5 * P0 * (1 - erf(np.sqrt(2) * (x - x0) / w0))


# -----------------------------
# Curve Fitting
# -----------------------------

# Initial parameter guesses: [max power, center position, beam waist]
initial_guess = [max(P), 4.5, 1.0]

# Perform weighted least-squares fit
popt, pcov = curve_fit(
    knife_edge,
    x,
    P,
    sigma=dP,
    p0=initial_guess,
    absolute_sigma=True
)

# Extract fitted parameters and uncertainties
P0_fit, x0_fit, w0_fit = popt
perr = np.sqrt(np.diag(pcov))  # standard deviations

# -----------------------------
# Beam Waist with Uncertainty
# -----------------------------

# Statistical uncertainty from fit
w0_fit_u = ufloat(w0_fit, perr[2])

# Instrument-limited uncertainty (micrometer resolution = 0.01 mm)
resolution = 0.01  # mm

# Combine uncertainties (recommended)
w0 = ufloat(w0_fit_u.n, np.sqrt(w0_fit_u.s**2 + resolution**2))

# -----------------------------
# Print Results
# -----------------------------

print(f"P0 = {P0_fit:.4f} ± {perr[0]:.4f} W")
print(f"x0 = {x0_fit:.4f} ± {perr[1]:.4f} mm")
print(f"w0 (fit only) = {w0_fit:.4f} ± {perr[2]:.4f} mm")
print(f"Beam waist w0 = {w0:.1uS} mm")  # nicely formatted

# -----------------------------
# Plot Data and Fit
# -----------------------------

# Generate smooth curve for plotting
x_fit = np.linspace(min(x), max(x), 400)
P_fit = knife_edge(x_fit, *popt)

plt.figure(figsize=(7, 5))

# Plot experimental data with error bars
plt.errorbar(x, P, yerr=dP, fmt='o', capsize=3, label='Data')

# Plot fitted curve
plt.plot(x_fit, P_fit, 'r-', label='Fit')

# Labels and formatting
plt.xlabel('Position (mm)')
plt.ylabel('Power (W)')
plt.title('Knife-edge Measurement of Beam Waist')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
