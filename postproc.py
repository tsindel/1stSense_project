import numpy as np
import matplotlib.pyplot as plt

x_refinement, p_refinement = 10, 5

res = np.load('result.pkl.npy')
p0, x0 = res[0,0], res[0,1]
col_ratios = list(np.round(np.linspace(p0, 1, p_refinement) * 100, 1))
nhw_ratio = np.linspace(x0, 1, x_refinement)
res_pp = np.reshape(res[:,2], (p_refinement,-1)).T

plt.plot(nhw_ratio * 100, res_pp * 100)
plt.title('Satellite collision mitigation results')
plt.xlabel('Ratio of sats with our HW (%)'), plt.ylabel('Collision reduction (%)')
plt.legend(col_ratios, title='Initial collision ratios (%)')
plt.grid()
plt.show()
