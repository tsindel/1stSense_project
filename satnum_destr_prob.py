import math
import numpy as np
import matplotlib.pyplot as plt

# Define constants
N = 1000 #  size of the sat fleet
Nd = 50  # size of the sat fleet with detection HW
Pd = 0.9  # probability of detection (true positive)
flux_total = 1e-2  # cumulative flux of debris in 1-10 mm range over all altitudes (#/m2/yr)
spatial_density_LEO = 1  # space debris density in LEO (1/km3)
spatial_density_GEO = 1e-6  # space debris density in GEO (1/km3)
Psize = 0.34  # probability that the detected debris is 1-10 mm if > 1mm are considered
range_detect = 1.5  # km ... radar detection range

Pdet = Pd * spatial_density_LEO * Psize / (4/3 * math.pi * range_detect ** 3)  # prob that radar detects debris
PdetN = 1 - (1 - Pdet) ** Nd  # prob that at least one radar detects debris

x = 5.5e-5 #  probability of one sat being destroyed in 1 year in orbit (1)
Pavoid = 0.8  # probability of a successful avoidance maneuver
Pdata = PdetN

xnew = x * (1 - Pavoid) * (1 - Pdata)
r = 0.04 #  display range ratio

M = np.arange(0, math.floor(N * r))
probs = np.array([math.comb(N, m) * x ** m * (1-x) ** (N-m) for m in M])
probs_new = np.array([math.comb(N, m) * xnew ** m * (1-xnew) ** (N-m) for m in M])

# Plot
plt.figure()
plt.xlabel('# of destroyed sats')
plt.ylabel('Probability (1 yr)')
plt.plot(M, probs)
plt.show()

print(x, xnew)
