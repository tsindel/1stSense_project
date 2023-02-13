import numpy as np
import matplotlib.pyplot as plt
import math

n_sats = 750

# Load data
p_ref_type, x_ref, p_ref = ['lin', 10, 5] #np.load('p_scale.pkl.npy')
x_refinement, p_refinement = int(x_ref), int(p_ref)
res = np.load("result_{}.pkl.npy".format(n_sats))

# Parse data
p0, p1, x0 = res[0, 0], res[-1, 0], res[0, 1]
p_scale = (
    np.linspace(p0, p1, p_refinement)
    if p_ref_type == "lin"
    else np.logspace(math.log10(p0), math.log10(p1), p_refinement)
)
col_ratios = list(p_scale)
cr = ["%.0E" % cr for cr in col_ratios]

nhw_ratio = np.linspace(x0, 1, x_refinement)
res_pp = np.reshape(res[:, 2], (p_refinement, -1)).T

plt.plot(nhw_ratio * 100, res_pp * 100)
plt.title("Satellite collision mitigation results")
plt.xlabel("Ratio of sats with our HW (%)"), plt.ylabel("Collision reduction (%)")
plt.legend(col_ratios, title="Initial collision ratios (1/yr)")
plt.grid()
plt.show()
