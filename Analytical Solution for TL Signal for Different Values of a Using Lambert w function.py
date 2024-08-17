import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wrightomega
import warnings
warnings.filterwarnings("ignore")

# Define the range of t values
t = np.arange(0, 1500.0, 1)

# Constants
N, R, s, E, kB, beta = 1.0E10, 0.5, 1e10, 1, 8.617e-5, 1

# Initial n0 value
n0 = 1e10

# Values of a to simulate
a_values = [2, 1, 0, -1, -2]

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Colors for different a values
colors = ['purple', 'red', 'green', 'orange', 'blue']

# Dictionary to store the area under the curve for each a
areas = {}

# Iterate over each value of a
for i, a in enumerate(a_values):
    c = ((n0 / N) * (1 - R) / R)
    x = E / (kB * ((t * beta) + 273))
    expint = ((E / kB) ** (a + 1)) / (x ** (a + 2)) * \
             np.exp(-E / (kB * (273 + t * beta))) * \
             (1 - ((a + 2) / x))
    zTL = (1 / c) - np.log(c) + (s * expint / (beta * ((1 - R))))
    lam = np.real(wrightomega(zTL))

    # Plot the analytical solution
    TL_signal = (N * R * ((273 + (beta * t)) ** a) / (beta * ((1 - R) ** 2.0))) * s * np.exp(-E / (kB * (273 + t * beta))) / (lam + lam ** 2)
    ax.plot(t, TL_signal, linewidth=2, label=f'$a = {a}$', color=colors[i])

    # Annotate the maximum point
    max_idx = np.argmax(TL_signal)
    max_t = t[max_idx]
    max_TL = TL_signal[max_idx]
    ax.annotate(f'Imax: {max_TL:.2e}\nTmax: {max_t:.6f}°C', xy=(max_t, max_TL), xytext=(max_t + 20, max_TL + 0.01 * max_TL),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='black')

    # Calculate the area under the curve using the trapezoidal rule
    area = np.trapz(TL_signal, t)
    areas[a] = area

# Add labels, legend, and title
ax.set_xlabel(r'Temperature T [$^{o}$C]')
ax.set_ylabel('TL signal [a.u.]')

# Adding the formula box
formula = (
    r"$I(T) = \frac{s_0 T^a N R \exp(-\frac{E}{k_B T})}{\beta (1-R)^2 \left(W\left[e^z(T)\right] + W\left[e^z(T)\right]^2\right)}$"
    "\n"
    r"$z = \frac{1}{c} - \ln(c) + \frac{s_0}{\beta (1-R)} \int_{T_0}^{T} T'^a \exp\left(-\frac{E}{k_B T'}\right) dT'$"
    "\n"
    r"$c = \frac{n_0 (1-R)}{N R}$"
    "\n"
    r"$\int_{T_0}^{T} T'^a \exp\left(-\frac{E}{k_B T'}\right) dT' = \frac{k_B T^{a+2}}{E} \left(1 - \frac{(a+2) k_B T}{E} + \frac{(a+2)(a+3)(k_B T)^2}{E^2} - \ldots \right)$"
)

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.691, 0.7, formula, transform=ax.transAxes, fontsize=10,
         verticalalignment='center', bbox=props)

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 0.99, 1])
plt.legend()
plt.title('Analytical Solution for TL Signal for Different Values of a Using Lambert w function')
plt.grid(True)
plt.show()

# Display the area under the curve for each a
print("Area under the curve for each value of a:")
for a, area in areas.items():
    print(f'a = {a}: Area = {area:.2e}')
