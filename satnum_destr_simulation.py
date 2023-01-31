import sat_debris_libs as sat
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sat_size = 5  # m
    det_rng = 0.8  # km
    sat_field = sat.generate_sats(1000, 3, sat_size, det_rng, 1200)
    deb_field = sat.generate_debris(1e-4,1200)
    interfs = sat.interfere(sat_field, deb_field, sat_size, det_rng)

    # Plot
    plt.figure(1)
    plt.title('Satellites')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Lattitude (km)')
    plt.scatter(sat_field[:,0], sat_field[:,1], sat_field[:,2] * 100)

    plt.figure(2)
    plt.title('Debris')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Lattitude (km)')
    plt.scatter(deb_field[:,0], deb_field[:,1], 1e-1)

    plt.show()


