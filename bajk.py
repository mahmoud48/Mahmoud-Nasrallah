import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the range for plotting
t = np.arange(5.0, 150.0, 1.0)

# Parameters for GOT model
no, N, R, s, E, kB, beta, a = 0.5E10, 1.0E10, 0.0000000000001, 9.182e17, 1.067, 8.617e-5, 1, -2
c = ((no / N) * (1 - R) / R)
x = E / (kB * ((t * beta) + 273))
expint = ((E / kB) ** (a + 1)) / (x ** (a + 2)) * \
        np.exp(-E / (kB * (273 + t * beta))) * \
        (1 - ((a + 2) / x))
zTL = (1 / c) - np.log(c) + (s * expint / (beta * ((1 - R))))
lam = np.real(lambertw(np.exp(zTL)))

# TL signal
TL_signal = 1000 * ((t * beta) ** a) * (N * R / ((1 - R) ** 2.0)) * s * np.exp(-E / (kB * (273 + t * beta))) / (lam + lam ** 2)

# Find the maximum value and corresponding temperature
max_intensity = np.max(TL_signal)
max_temp = t[np.argmax(TL_signal)]

# Plot TL signal
plt.plot(t, TL_signal, 'b--', label='TL x1000')
plt.ylabel('TL signal [a.u.]')
plt.xlabel(r'Temperature T [$^{o}$C]')

# Annotate the maximum point
plt.annotate(f'Max Intensity: {max_intensity:.2f}\nMax Temp: {max_temp:.2f}°C',
             xy=(max_temp, max_intensity), xycoords='data',
             xytext=(max_temp + 20, max_intensity + 200),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()

# Layout adjustments
plt.tight_layout()

# Show the plot
plt.show()

# Print the maximum temperature
print(f"Maximum Temperature: {max_temp}°C")
