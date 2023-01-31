# def sat_detect_algo(x_sats, y_sats, z_sats, n_sat_hw, threshold_det, sat_avg_size, irreg_ratio, altitude, n_debris):
#     threshold_col = irreg_ratio * sat_avg_size / 2e3

import numpy as np
import sat_debris_libs as sat
from tqdm.auto import tqdm

d = 40
n_debris = 5000
Nsat = 10
Nhw = 100
alt = 2000

x_sats, y_sats, z_sats = sat.gen_sat_constellation(Nsat, alt, spacing='random')  # generate sat constellation

# Generate random debris orbits
x_circle = y_circle = z_circle = np.random.uniform(-1000,1000,size=(d, n_debris))

# Compute the shortest distance from each satellite to the random orbit
n_sats = len(x_sats)
norms = np.empty((n_sats, n_debris))
for sat in tqdm(np.nditer(np.arange(n_sats)), desc='Calculating interferences', total=n_sats, position=0):
    for deb in tqdm(np.nditer(np.arange(n_debris)), total=n_debris, position=1):
        # sat, deb = sat.astype(np.int64), deb.astype(np.int64)
        diffs = np.column_stack((x_sats[sat], y_sats[sat], z_sats[sat])) - np.column_stack(
            (x_circle[:, deb], y_circle[:, deb], z_circle[:, deb])
        )
        norms[sat, deb] = np.min(np.linalg.norm(diffs, axis=1))

hw = np.random.choice(np.arange(n_sats), Nhw)  # pick indexes of sats with hardware randomly
# debri_det = get_debris_candidates(norms, n_sats, n_debris, hw, threshold_det, threshold_col)  # get indexes of debris
#
# # Remove detected debris from collision potential
# norms_det = np.delete(norms, debri_det, axis=1)
#
# # get indexes of all sats in collision range
# sats_in_collision_before = np.where(norms <= threshold_col)
# sats_in_collision_after = np.where(norms_det <= threshold_col)
#
# # get the sat collision numbers
# n_sats_col_before = sats_in_collision_before[0].shape[0]
# n_sats_col_after = sats_in_collision_after[0].shape[0]
#
# return n_sats_col_before, n_sats_col_after