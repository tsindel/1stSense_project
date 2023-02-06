import math
import numpy as np
from tqdm.auto import tqdm
from scipy.special import comb
import asyncio


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def gen_planet(altitude_km, r_earth=6378):
    # Generate data for the sphere
    theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    THETA, PHI = np.meshgrid(theta, phi)

    # Create a smaller sphere with Earth-like texture
    r = r_earth
    x = r * np.sin(PHI) * np.cos(THETA)
    y = r * np.sin(PHI) * np.sin(THETA)
    z = r * np.cos(PHI)

    # Create orbit surface (circular approx)
    R = r_earth + altitude_km
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    return X, Y, Z, x, y, z


def gen_sat_constellation(n_sat, altitude_km, spacing="random"):
    earth_rad = 6378  # radius of Earth's surface in km
    R = earth_rad + altitude_km

    # Generate N points on the sphere
    if spacing == "uniform":
        m_int = math.ceil(
            np.sqrt(n_sat) + 1
        )  # math.ceil(1/4 * (3 + np.sqrt(8 * n_sat + 1))) ... for double amount on longitude
        phi_south = np.pi / m_int
        phi_north = 2 * np.pi - phi_south
        sat_phi_ = np.tile(np.linspace(phi_south, phi_north, m_int), 1 * m_int - 1)
        sat_phi = sat_phi_[0:n_sat]
        sat_theta = np.zeros(n_sat)
        for meridian in np.arange(1 * m_int - 1):
            idx_sta = meridian * (m_int - 1)
            idx_fin = meridian * (m_int - 1) + m_int - 2
            sat_theta[idx_sta:idx_fin] = np.pi / m_int * meridian
    elif spacing == "random":
        sat_phi = np.random.uniform(0, np.pi, n_sat)
        sat_theta = np.random.uniform(0, 2 * np.pi, n_sat)
    else:
        print('Invalid spacing parameter. Please specify either "uniform" or "random".')

    # Calculate x, y, z coordinates for the points
    x_sats = R * np.sin(sat_phi) * np.cos(sat_theta)
    y_sats = R * np.sin(sat_phi) * np.sin(sat_theta)
    z_sats = R * np.cos(sat_phi)

    return x_sats, y_sats, z_sats


@background
def gen_rand_orbits(altitude, num_orbits=1, r_earth=6378, discr=40):
    R = r_earth + altitude

    # Generate random angles for rotation
    random_x_angle = np.random.uniform(0, 2 * np.pi, size=(num_orbits,))
    random_y_angle = np.random.uniform(0, 2 * np.pi, size=(num_orbits,))
    random_z_angle = np.random.uniform(0, 2 * np.pi, size=(num_orbits,))

    x_orbits = y_orbits = z_orbits = np.empty((discr,))
    for i in tqdm(
        np.arange(num_orbits), desc="Generating debris trajectories", total=num_orbits
    ):
        theta = np.linspace(0, 2 * np.pi, discr)

        # Create rotation matrices
        rx = np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(random_x_angle[i]), -np.sin(random_x_angle[i])],
                [0, np.sin(random_x_angle[i]), np.cos(random_x_angle[i])],
            ]
        )

        ry = np.matrix(
            [
                [np.cos(random_y_angle[i]), 0, np.sin(random_y_angle[i])],
                [0, 1, 0],
                [-np.sin(random_y_angle[i]), 0, np.cos(random_y_angle[i])],
            ]
        )

        rz = np.matrix(
            [
                [np.cos(random_z_angle[i]), -np.sin(random_z_angle[i]), 0],
                [np.sin(random_z_angle[i]), np.cos(random_z_angle[i]), 0],
                [0, 0, 1],
            ]
        )

        # Generate points for the circle
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = np.zeros(discr)

        points = np.asarray(np.column_stack((x, y, z)) * rx * ry * rz)

        x_orbits = np.column_stack((x_orbits, points[:, 0]))
        y_orbits = np.column_stack((y_orbits, points[:, 1]))
        z_orbits = np.column_stack((z_orbits, points[:, 2]))

    return (
        x_orbits,
        y_orbits,
        z_orbits,
    )  # return 2D arrays of orbit coords for each debris


@background
def sat_2_orb_dists(
    x_sats,
    y_sats,
    z_sats,
    x_circle,
    y_circle,
    z_circle,
    n_sat_hw,
    threshold_det,
    threshold_col=5e-3,
):

    # Compute the shortest distance from each satellite to the random orbit
    norms = np.load("norms.pkl.npy")  # 'norms' file must be saved before executing this function
    n_sats = len(x_sats)
    distances = np.zeros(n_sats)
    for i in tqdm(np.arange(n_sats), desc="Calculating interferences", total=n_sats):
        diffs = np.column_stack((x_sats[i], y_sats[i], z_sats[i])) - np.column_stack(
            (x_circle.T.flatten(), y_circle.T.flatten(), z_circle.T.flatten())
        )
        norms = np.linalg.norm(diffs, axis=1)
        distances[i] = np.min(norms)  # THIS ASSUMES THAT ONE COLLISION IS ENOUGH TO DESTROY THE SATELLITE

    distances_hw = np.random.choice(
        distances, size=n_sat_hw
    )  # we use random choice of HW equipped sats for now

    sats_in_detection = np.where(
        distances_hw <= threshold_det
    )  # get indexes of sats in sats with hw in detection range
    n_sats_det = len(sats_in_detection[0])

    sats_in_collision = np.where(
        distances <= threshold_col
    )  # get indexes of all sats in collision range
    n_sats_col = len(sats_in_collision[0])

    min_dist = np.min(distances)

    return n_sats_det, n_sats_col, sats_in_detection, sats_in_collision, min_dist


@background
def optimize_collisions(
    n_sat_collisions, sat_avg_size, norms,
    max_iters = 500
):

    """This fcn computes the required irregularity ratio of the sats to result in the specified number
    of collisions annually"""

    # Compute the number of orbits needed to ensure that n_sat_det sats detect these orbits
    irreg_ratio, iter = 0, 0  # sat shape irregularity ratio (measures debris collision exposure)
    pbar = tqdm()
    while iter <= max_iters:
        iter += 1
        irreg_ratio += 10
        rcol = irreg_ratio * sat_avg_size / 2e3
        n_sats_col = np.count_nonzero(norms <= rcol)
        # print('Rcol = {},\tCollisions = {}'.format(rcol, n_sats_col))

        if n_sats_col >= n_sat_collisions:
            break

        pbar.update()

    pbar.close()
    print(
        "{} irregularity ratio needed to have {} collisions per year:".format(
            irreg_ratio, n_sat_collisions
        )
    )

    return irreg_ratio


@background
def get_debris_candidates(norms, hw, rdet, rcol):

    """najde úlomky které jsou zároveň detekovány
    a zároveň kolizně ohrožují jiný satelit"""

    debris_candidates, d = (
        np.empty(
            0,
        ),
        norms.shape[1],
    )
    pbar = tqdm(total=d, desc="Analyzing debris candidates", position=0, leave=False)
    for debris in np.arange(d):
        detected = np.intersect1d(
            hw, np.where(norms[:, debris] <= rdet)[0]
        )  # find sats with HW which had this debris in detection range
        collided = np.where(norms[:, debris] <= rcol)[
            0
        ]  # find sats (idx) which collided with this debris
        debris_candidates = (
            np.append(debris_candidates, debris)
            if not (detected.size == 0 or collided.size == 0)
            else debris_candidates
        )
        # mark this debris as avoidance candidate if it was both detected and collided
        pbar.update()

    pbar.close()

    print("There is {} debris candidates".format(len(debris_candidates)))

    return debris_candidates


@background
def get_distances_file(
    x_sats,
    y_sats,
    z_sats,
    x_circle,
    y_circle,
    z_circle,
    n_debris,
):
    n_sats = x_sats.shape[0]
    # Compute the shortest distance from each satellite to the random orbit
    norms = np.empty((n_sats, n_debris))
    for sat in tqdm(
        np.arange(n_sats),
        desc="Calculating interferences",
        total=n_sats,
        position=0,
        leave=True,
    ):
        for deb in tqdm(
            np.arange(n_debris),
            desc="Sat {}/{}".format(sat + 1, n_sats),
            total=n_debris,
            position=0,
            leave=True,
        ):
            diffs = np.column_stack(
                (x_sats[sat], y_sats[sat], z_sats[sat])
            ) - np.column_stack(
                (x_circle[:, deb + 1], y_circle[:, deb + 1], z_circle[:, deb + 1])
            )
            norms[sat, deb] = np.min(np.linalg.norm(diffs, axis=1))
    np.save("norms.pkl", norms)


def sat_detect_algo(
    hw,
    threshold_det,
    sat_avg_size,
    irreg_ratio,
    norms,
):
    threshold_col = irreg_ratio * sat_avg_size / 2e3

    # get indexes of debris avoidance candidates
    debris_det = get_debris_candidates(norms, hw, threshold_det, threshold_col).astype(int)

    # Remove detected debris from collision potential
    norms_det = np.delete(norms, debris_det, axis=1)

    # get the sat collision numbers
    n_sats_col_before = np.count_nonzero(norms <= threshold_col)
    n_sats_col_after = np.count_nonzero(norms_det <= threshold_col)

    return n_sats_col_before, n_sats_col_after


def binom_prob(N, m, x):
    return comb(N, m) * np.power(x, m) * np.power(1 - x, N - m)


def find_max_prob(group_elems: object, unit_prob: object) -> object:
    m_vals = np.arange(group_elems + 1)
    probs = binom_prob(group_elems, m_vals, unit_prob)
    max_index = np.argmax(probs)
    return m_vals[max_index], probs[max_index]
