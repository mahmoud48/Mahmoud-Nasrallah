import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Initial conditions.
n0, nc0, m0 = 1e10, 0, 1e10
# Initial conditions vector
y0 = n0, nc0, m0
# Numerical parameters for OTOR.
N, An, Am, s, E, hr = 1e10, 0.5e-8, 1e-8, 1e10, 1, 1

# A grid of time points in seconds
t = np.linspace(0, 880, 880)
kB = 8.617e-5

# Values of 'a' to iterate over
a_values = [-2, -1, 0, 1, 2]

# Plotting setup
plt.figure(figsize=(14, 10))

for a in a_values:
    # Differential equations of the OTOR model.
    def deriv(y, t):
        n, nc, m = y
        dndt = - n * s * ((273 + (t * hr)) ** a) * np.exp(-E / (kB * (273 + hr * t))) + nc * An * (N - n)
        dncdt = n * s * ((273 + (t * hr)) ** a) * np.exp(-E / (kB * (273 + hr * t))) - nc * An * (N - n) - m * Am * nc
        dmdt = -m * Am * nc
        return dndt, dncdt, dmdt

    # Integrate the OTOR equations over the time grid, t.
    # Call `odeint` to generate the solution.
    ret = odeint(deriv, y0, t)
    n, nc, m = ret.T

    # Compute TL signal
    TL = n * s * ((273 + (t * hr)) ** a) * np.exp(-E / (kB * (273 + hr * t))) - nc * An * (N - n)

    # Plot the TL signal
    plt.plot(hr * t, TL, linewidth=2, label=f'a = {a}')

    # Find the peak of the TL signal
    max_intensity = np.max(TL)
    max_index = np.argmax(TL)
    max_temperature = hr * t[max_index]

    # Annotate the peak
    plt.annotate(
        f'Max: {max_intensity:.2e}\nT: {max_temperature:.1f}°C',
        xy=(max_temperature, max_intensity),
        xytext=(max_temperature + 20, max_intensity + max_intensity * 0.01),
        arrowprops=dict(facecolor='black', arrowstyle='->')
    )

# Adding the formula box
formula = (
    r"$\frac{dn_c}{dt} = n s_0 T^a e^{-\frac{E}{k_B T}} - A_n (N-n) n_c - A_m n_c m$"
    "\n"
    r"$\frac{dn}{dt} = A_n (N-n) n_c - n s_0 T^a e^{-\frac{E}{k_B T}}$"
    "\n"
    r"$I = -\frac{dm}{dt} = A_m n_c m$"
)

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(.742, 0.7, formula, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='center', bbox=props)

# Final plot adjustments
plt.ylabel('TL signal [a.u.]')
plt.xlabel(r'Temperature T [$^{o}$C]')
plt.title('Numerical Solution for TL Signal for Different Values of a Using rate equations', fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.99, 1])
plt.show()