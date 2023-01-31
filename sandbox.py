import sat_debris_libs as sat
import numpy as np
import matplotlib.pyplot as plt

Nsat = 1000  # total number in the sat constellation
sat_destr_prob = 0.55 / 100  # 0.55 %

n_colls_data = sat.find_max_prob(Nsat, sat_destr_prob)[0]
print(n_colls_data)

m_vals = np.arange(Nsat + 1)
probs = sat.binom_prob(Nsat, m_vals, sat_destr_prob)

plt.plot(m_vals, probs)
plt.show()