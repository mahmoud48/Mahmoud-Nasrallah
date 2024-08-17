import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the range for plotting
t = np.arange(0.0, 150.0, 1.0)



# Parameters for GOT model
no, N, R, s, E, kB, beta, a = 1.0E10, 1.0E10, 0.5, 1e10, 1, 8.617e-5, 1, 2
c = ((no / N) * (1 - R) / R)
x = E / (kB * ((t * beta) + 273))
expint = ((E / kB) ** (a + 1)) / (x ** (a + 2)) * \
        np.exp(-E / (kB * (273 + t * beta))) * \
        (1 - ((a + 2) / x))
zTL = (1 / c) - np.log(c) + (s * expint / (beta * ((1 - R))))
lam = np.real(lambertw(np.exp(zTL)))

# TL signal plot
plt.plot(t, (N * R / ((1 - R))) * (1 / lam), 'r-', label='n(t)')
plt.xlabel('Temperature T [$^{o}$C]')
plt.ylabel('n(t) and TL')


# Plot TL signal
plt.plot(t, 1000* ((t * beta) ** a) * (N * R / ((1 - R) ** 2.0)) * s * np.exp(-E / (kB * (273 + t * beta))) / (lam + lam ** 2), 'b--', label='TL x1000')
plt.ylabel('TL signal [a.u.]')
plt.xlabel(r'Temperature T [$^{o}$C]')
plt.legend()


# Layout adjustments
plt.tight_layout()

# Show the plot
plt.show()
