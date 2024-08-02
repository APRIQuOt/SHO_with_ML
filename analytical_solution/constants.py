import numpy as np

hbar = 1.0545718e-34  # Reduced Planck constant in J.s
m1 = 1.0  # Mass of oscillator 1
m2 = 1.0  # Mass of oscillator 2
omega1 = 1.0  # Frequency of oscillator 1
omega2 = 4  # Frequency of oscillator 2
k1, k2, k3, k4 = 0.1, 0.4, 0.3, 0.5  # Coupling constants

# Interaction coefficients
A12 = hbar * k1 * np.sqrt(m2 * omega2 / (m1 * omega1))
A21 = hbar * k2 * np.sqrt(m1 * omega1 / (m2 * omega2))
B = hbar * k3 * np.sqrt(1 / (m1 * omega1 * m2 * omega2))
C = hbar * k4 * np.sqrt(1 / (m1 * omega1 * m2 * omega2))

# Derived quantities
Omega = (1 / hbar) * np.sqrt((B - C) ** 2 + (A12 - A21) ** 2)
epsilon = (omega2 - omega1) / Omega


t = 1.0  # Time
s1, s2 = 1, 1  # Initial quantum numbers
n = 0  
m = s1 + s2 - n 