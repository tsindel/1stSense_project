import numpy as np
import matplotlib.pyplot as plt
import math

x_refinement, p_refinement = 10, 6
p_ref_type = 'log'  # log / lin

res = np.load('result.pkl.npy')
p0, p1, x0 = res[0,0], res[-1,0], res[0,1]
p_scale = np.linspace(p0, p1, p_refinement) if p_ref_type == 'lin' else np.logspace(math.log10(p0), math.log10(p1), p_refinement)
col_ratios = list(p_scale)
cr = ['%.0E' % cr for cr in col_ratios]

nhw_ratio = np.linspace(x0, 1, x_refinement)
res_pp = np.reshape(res[:,2], (p_refinement,-1)).T

plt.plot(nhw_ratio * 100, res_pp * 100)
plt.title('Satellite collision mitigation results')
plt.xlabel('Ratio of sats with our HW (%)'), plt.ylabel('Collision reduction (%)')
plt.legend(col_ratios, title='Initial collision ratios (1/yr)')
plt.grid()
plt.show()
