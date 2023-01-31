# Define constants
import math
import numpy as np
import matplotlib.pyplot as plt

Nhw = 50  # size of the sat fleet with detection HW
Pd = 0.9  # probability of detection (true positive)
flux_total = 1.367 / (1e-2 - 1e-3)  # cumulative flux of debris in 1-10 mm range over all altitudes (#/km2/yr)
spatial_density_LEO = 1  # space debris density in LEO (1/km3)
spatial_density_GEO = 1e-6  # space debris density in GEO (1/km3)
Psize = 0.69  # probability that the detected debris is 1-10 mm if > 1mm are considered
range_detect = 1.5  # km ... radar detection range

Nratio = 0.3  # ratio of Nflux axis scaling

Nflux = round(flux_total * math.pi * range_detect ** 2)  # number of detected dabris 1-10 mm / yr
nfluxes = np.arange(round(Nflux * 0.8), round(Nflux * 1))

probs_det = np.array([math.comb(Nflux, nflux) * Pd ** nflux * (1 - Pd) ** (Nflux - nflux) for nflux in nfluxes])

# Pdet = Pd * spatial_density_LEO * Psize / (4/3 * math.pi * range_detect ** 3)  # prob that radar detects debris
# PdetN = 1 - (1 - Pdet) ** N  # prob that at least one radar detects debris

# Plot
plt.figure()
plt.xlabel('Ratio of detected debris per year (%)')
plt.ylabel('Probability with given # of space debris')
plt.plot(100 * nfluxes / Nflux, probs_det)
plt.show()

