from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")
from scipy.special import wrightomega

# Load the data
data = np.loadtxt('https://raw.githubusercontent.com/vpagonis/Python-Codes/main/Ch3PagonisGitHub/Refglow009.txt')
x_data, y_data = data[:, 0], data[:, 1]

# Constants
kB = 8.617E-5
a = 0

# Define the TL function
def TL(T, imax, R, E, Tmax):
    x = E / (kB * T)
    xm = E / (kB * Tmax)
    F = kB * (T ** (2.0 + a)) * np.exp(-E / (kB * T)) * (1 - ((a + 2) / x) + (((a + 2) * (a + 3)) / (x ** 2)) - (((a + 2) * (a + 3) * (a + 4) / (x ** 3))))
    Fm = kB * (Tmax ** (a + 2)) * np.exp(-E / (kB * Tmax)) * (1 - ((a + 2) / xm) + (((a + 2) * (a + 3)) / (xm ** 2)) - (((a + 2) * (a + 3) * (a + 4) / (xm ** 3))))
    b = ((a / (Tmax ** (a + 1))) + (E) / (kB * (Tmax ** (a + 2)))) / (1 - 1.05 * (R ** (1.26)))
    Z = R / (1 - R) - np.log((1 - R) / R) + (F * np.exp(E / (kB * Tmax))) * b / E
    Zm = R / (1 - R) - np.log((1 - R) / R) + (Fm * np.exp(E / (kB * Tmax))) * b / E
    argW = wrightomega(Z)
    argWm = wrightomega(Zm)
    return imax * ((T / Tmax) ** a) * np.exp(-E / (kB * T) * (Tmax - T) / Tmax) * (argWm + argWm ** 2.0) / (argW + argW ** 2.0)

# Define the total TL function
def total_TL(T, *inis):
    u = np.array([0 for i in range(len(x_data))])
    imaxs, Rs, Es, Tmaxs = inis[0:9], inis[9:18], inis[18:27], inis[27:36]
    for i in range(9):
        u = u + TL(T, imaxs[i], Rs[i], Es[i], Tmaxs[i])
    return u

# Initial parameters
inis = (9824,21009,27792,50520,7153,5496,6080,1641,2316,
0.01,0.01,.01,.01,.01,.01,.01,.01,.01,
1.20,1.32,2.06, 2.61,1.39, 1.11,2.43,2.925,2.21,
387,428,462,488,493,528,559,585, 602)
# Fit the data
params, params_covariance = optimize.curve_fit(total_TL, x_data, y_data, p0=inis)

# Plotting
plt.subplot(2, 1, 1)
plt.scatter(x_data, y_data, label='Experiment')
plt.plot(x_data, total_TL(x_data, *params), c='r', linewidth=3, label='Equation (2.58), a = 1')

# Plot and annotate individual peaks
for i in range(9):
    peak_TL = TL(x_data, *params[i:36:9])
    plt.plot(x_data, peak_TL)
    Tmax = params[27 + i]
    imax = params[i]


leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.ylabel('TL signal [a.u.]')
plt.xlabel(r'Temperature T [K]')
plt.text(350, 58000, 'TLD-700')
plt.text(350, 50000, 'GLOCANIN')
plt.text(350, 42000, 'Refglow#9')

plt.subplot(2, 1, 2)
plt.scatter(x_data, total_TL(x_data, *params) - y_data, c='r', linewidth=2, label='Residuals')
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.ylabel('Residuals')
plt.xlabel(r'Temperature T [K]')
plt.ylim(-20000, 20000)
plt.tight_layout()

# Calculate and print FOM
res = total_TL(x_data, *params) - y_data
FOM = round(100 * np.sum(abs(res)) / np.sum(y_data), 2)
myTable = PrettyTable(["Imax", "R", "E (eV)", "Tmax (K)"])
for i in range(9):
    myTable.add_row(np.round(params[i:36:9], 2))

print('FOM=', FOM, ' %')
print(myTable)
plt.show()
