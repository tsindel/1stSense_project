import matplotlib.pyplot as plt
import sat_debris_libs as sat
import numpy as np
from tqdm.auto import tqdm


########################################################################################################################
"""Simulation parameter definition"""
altitude = 2000  # orbit altitude in km
Nsat = 10000  # total number in the sat constellation
sat_destr_prob = 0.55 / 100  # 0.55 %
Nhw = 100  # default number of hardware-equipped sats (our hardware)
detection_range = 0.8  # detection range of the hardware in km
sat_avg_size = 5  # average satellite size in m
n_debris = 50000  # number of pieces of debris in the given orbit (scientific data)
sat_spacing = "uniform"  # 'random' or 'uniform' spacing of sats
irreg_ratio_custom = 132  # custom irregularity ratio for sat-debris interactions
x_refinement = 10
p_refinement = 6   # number of different collision ratios considered in graph
optimize = True  # Chooses whether to optimize irregularity ratio for sats

########################################################################################################################
"""Generate model of planet Earth with specified orbit"""
print('\nGenerating orbit model...')
X, Y, Z, x, y, z = sat.gen_planet(
    altitude
)  # generate planet Earth (Yes, I am a God now)
x_sats, y_sats, z_sats = sat.gen_sat_constellation(
    Nsat, altitude, spacing=sat_spacing
)  # generate sat constellation

"""Generate space debris with random orbits"""
try:
    print('\nLoading space debris trajectories...')
    x_circle, y_circle, z_circle = np.load("xout.pkl.npy"), np.load("yout.pkl.npy"), np.load("zout.pkl.npy")
except FileNotFoundError:
    print('\nGenerating space debris trajectories...')
    x_circle, y_circle, z_circle = sat.gen_rand_orbits(altitude, n_debris)
    np.save("xout.pkl", x_circle), np.save("yout.pkl", x_circle), np.save(
        "zout.pkl", x_circle
    )

"""Calculate all distances of sat-debris combinations and save them to file"""
try:
    print('\nLoading orbital distances...')
    norms = np.load("norms.pkl.npy")
except FileNotFoundError:
    print('\nComputing orbital distances...')
    sat.get_distances_file(x_sats, y_sats, z_sats, x_circle, y_circle, z_circle, n_debris)
    norms = np.load("norms.pkl.npy")  # load distances saved to file by previous fcn

"""Calibrate irregurality ratio of sat-debris interaction with respect to data"""
if optimize:
    print('\nCalibrating debris irregularity ratio...')
    n_cols_data = sat.find_max_prob(Nsat, sat_destr_prob)[0]
    irreg_ratio_0 = sat.optimize_collisions(
        n_cols_data, sat_avg_size, norms
    )
else:
    irreg_ratio_0 = irreg_ratio_custom

"""Analyze different detection & collision configurations"""
result = np.empty((0,3),int)
for coll_ratio in tqdm(np.logspace(-6, -1, p_refinement), total=p_refinement):
    """Optimize irregurality ratio of sat-debris interaction"""
    n_colls_desired = round(Nsat * coll_ratio)
    if optimize:
        print('\nOptimizing debris irregularity ratio...')
        irreg_ratio = sat.optimize_collisions(
            n_colls_desired, sat_avg_size, norms
        )
    else:
        irreg_ratio = irreg_ratio_custom

    """Simulate collisions with and w/o detection & avoidance hardware"""
    for nhw_ratio in tqdm(np.linspace(0.1,1,x_refinement), total=x_refinement):
        print('\nSimulating collisions for {} #HW ratio...'.format(nhw_ratio))
        nhw = round(nhw_ratio * Nsat)
        det_calib = irreg_ratio / irreg_ratio_0  # calibrating detection range based on debris irregularity
        cols_nohw, cols_hw = sat.sat_detect_algo(
            nhw,
            detection_range * det_calib,
            sat_avg_size,
            irreg_ratio,
            norms,
        )

        # Calculate debris collision reduction
        col_reduction = 0 if cols_nohw == 0 else (cols_nohw - cols_hw) / cols_nohw

        result = np.append(result, [[coll_ratio, nhw_ratio, col_reduction]], axis=0)  # append result to table

np.save('result.pkl', result)  # save the table of results

print(
    "{} collisions without system, {} collisions with system\nCollisions reduced by {} %".format(
        cols_nohw, cols_hw, col_reduction * 100
    )
)

"""Plot the Earth and given orbit with satellites"""
# Plot the sphere
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(X, Y, Z, color="b", linewidth=0.5)
ax.view_init(elev=30, azim=30)

ax.plot_surface(x, y, z, color="blue", alpha=0.5)
ax.plot_surface(x, y, -z, color="white", alpha=0.5)

# Plot the 3 points
ax.scatter(x_sats, y_sats, z_sats, color="r")

# Adding an axis of the sphere in red color
R = 6378 + altitude
ax.quiver(0, 0, 0, 0, 0, +1.3 * R, color="r", arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, -1.3 * R, color="r", arrow_length_ratio=0.1)

# Add labels
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")

plt.show()
