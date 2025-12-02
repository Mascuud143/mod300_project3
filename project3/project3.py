import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



class Sphere:
    def __init__(self, center_x, center_y, center_z, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius

class Box:
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

def plot_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs, c='r', marker='o', s=1)
    ax.set_title(title)
    plt.show()


def generate_random_point_in_box(box):
    x = random.uniform(box.min_x, box.max_x)
    y = random.uniform(box.min_y, box.max_y)
    z = random.uniform(box.min_z, box.max_z)
    return np.array([x, y, z])


def is_point_inside_sphere(point, sphere):
    dist_sq = ((point[0] - sphere.center_x) ** 2 +
               (point[1] - sphere.center_y) ** 2 +
               (point[2] - sphere.center_z) ** 2)
    return dist_sq < sphere.radius ** 2


def calculate_and_plot_fraction_sphere(sphere,box, N):
    inside_count = 0
    x_plot, y_plot, z_plot = [], [], []
    for _ in range(N):
        point = generate_random_point_in_box(box)
        if is_point_inside_sphere(point, sphere):
            inside_count += 1
            x_plot.append(point[0])
            y_plot.append(point[1])
            z_plot.append(point[2])
    fraction_inside = inside_count / N
    plot_points(list(zip(x_plot, y_plot, z_plot)), f'Points inside sphere (N={N})')
    return fraction_inside


def plot_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs, c='r', marker='o', s=1)
    ax.set_title(title)
    plt.show()



def pi_calculation(N):
    inside_circle = 0
    for _ in range(N):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    pi_estimate = (inside_circle / N) * 4
    return pi_estimate

def generate_spheres_in_box(num_spheres, box):
    spheres = []
    for _ in range(num_spheres):
        radius = random.uniform(0.1, 0.3)
        center_x = random.uniform(box.min_x + radius, box.max_x - radius)
        center_y = random.uniform(box.min_y + radius, box.max_y - radius)
        center_z = random.uniform(box.min_z + radius, box.max_z - radius)
        spheres.append(Sphere(center_x, center_y, center_z, radius))
    
    return spheres


def plot_fraction_spheres(spheres, box, N):
    x_plot, y_plot, z_plot = [], [], []
    inside_point_count = 0
    for _ in range(N):
        random_point = generate_random_point_in_box(box)
        for sphere in spheres:
            if is_point_inside_sphere(random_point, sphere):
                x_plot.append(random_point[0])
                y_plot.append(random_point[1])
                z_plot.append(random_point[2])
                inside_point_count += 1
                break

    fraction_inside = inside_point_count / N
    estimated_volume = fraction_inside * ((box.max_x - box.min_x) * (box.max_y - box.min_y) * (box.max_z - box.min_z))
    exact_volume = sum((4/3) * math.pi * sphere.radius**3 for sphere in spheres)

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x_plot, y_plot, z_plot, c='r', marker='o', s=1)
    ax.set_title(f'Points inside spheres (N={N})')
    plt.show()

    return fraction_inside, estimated_volume, exact_volume


def plot_fraction_dna(dna_spheres, box, N_points=100000):
    inside_count = 0
    x_in, y_in, z_in = [], [], []

    for _ in range(N_points):
        point = generate_random_point_in_box(box)
        for (center_x, center_y, center_z, radius) in dna_spheres:
            sphere = Sphere(center_x, center_y, center_z, radius)
            if is_point_inside_sphere(point, sphere):
                inside_count += 1
                x_in.append(point[0])
                y_in.append(point[1])
                z_in.append(point[2])
                break  # stop after the first sphere that contains the point

    fraction_inside = inside_count / N_points
    box_volume = (box.max_x - box.min_x) * (box.max_y - box.min_y) * (box.max_z - box.min_z)
    estimated_volume_dna = fraction_inside * box_volume

    # exact sum of atomic sphere volumes (ignores overlap)
    exact_volume_dna = sum((4 / 3) * math.pi * (radius ** 3) for (_, _, _, radius) in dna_spheres)

    # Plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x_in, y_in, z_in, c='g', marker='o', s=1, alpha=0.2)
    ax.set_title(f'Points Inside DNA Molecule (N={N_points})')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

    return fraction_inside, estimated_volume_dna, exact_volume_dna



def generate_random_walkers(num_walkers, box, n_steps=1000, step_size=1.0):

    walkers = []
    for _ in range(num_walkers):
        x, y, z = generate_random_point_in_box(box)
        path_x, path_y, path_z = [x], [y], [z]

        for _ in range(n_steps):
            # Random step in 3D
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            dz = random.uniform(-step_size, step_size)

            # Update position
            x += dx
            y += dy
            z += dz

            # Keep inside box boundaries
            x = max(box.min_x, min(box.max_x, x))
            y = max(box.min_y, min(box.max_y, y))
            z = max(box.min_z, min(box.max_z, z))

            # Store position
            path_x.append(x)
            path_y.append(y)
            path_z.append(z)

        walkers.append({'path_x': path_x, 'path_y': path_y, 'path_z': path_z})

    return walkers



def generate_random_walkers_fast(num_walkers, box, n_steps=1000, step_size=1.0):

    # Initial positions (num_walkers, 3)
    positions = np.random.uniform(
        low=[box.min_x, box.min_y, box.min_z],
        high=[box.max_x, box.max_y, box.max_z],
        size=(num_walkers, 3)
    )
    # store paths
    paths_x = np.empty((num_walkers, n_steps + 1), dtype=float)
    paths_y = np.empty((num_walkers, n_steps + 1), dtype=float)
    paths_z = np.empty((num_walkers, n_steps + 1), dtype=float)

    paths_x[:, 0] = positions[:, 0]
    paths_y[:, 0] = positions[:, 1]
    paths_z[:, 0] = positions[:, 2]

    # Walk
    for t in range(1, n_steps + 1):
        # uniform steps for all walkers (num_walkers, 3)
        steps = np.random.uniform(-step_size, step_size, size=(num_walkers, 3))
        positions += steps

        # keep inside box
        positions[:, 0] = np.clip(positions[:, 0], box.min_x, box.max_x)
        positions[:, 1] = np.clip(positions[:, 1], box.min_y, box.max_y)
        positions[:, 2] = np.clip(positions[:, 2], box.min_z, box.max_z)

        # store
        paths_x[:, t] = positions[:, 0]
        paths_y[:, t] = positions[:, 1]
        paths_z[:, t] = positions[:, 2]

    # Match your previous structure: list of dicts
    walkers = [
        {"path_x": paths_x[i], "path_y": paths_y[i], "path_z": paths_z[i]}
        for i in range(num_walkers)
    ]
    return walkers


def plot_random_walkers(walkers, title="Random Walkers Paths"):
    """
    Plot 3D paths of walkers returned by generate_random_walkers_fast.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for w in walkers:
        ax.plot(w["path_x"], w["path_y"], w["path_z"], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()



def test_strategy_one_sphere_random_walkers(
    box,
    sphere,
    n_walkers: int = 100,
    n_steps: int = 1000,
    step_size: float = 0.1
):
    """
    Returns:
        accessible_fraction, accessible_volume_est, sphere_volume_est, sphere_volume_exact
    """
    walkers = generate_random_walkers_fast(
        num_walkers=n_walkers, box=box, n_steps=n_steps, step_size=step_size
    )

    accessible_count = 0
    total_steps = len(walkers) * len(walkers[0]["path_x"])  # includes initial positions

    for w in walkers:
        for x, y, z in zip(w["path_x"], w["path_y"], w["path_z"]):
            # pass a point array + Sphere object
            if not is_point_inside_sphere(np.array([x, y, z]), sphere):
                accessible_count += 1

    accessible_fraction = accessible_count / total_steps
    box_volume = (
        (box.max_x - box.min_x)
        * (box.max_y - box.min_y)
        * (box.max_z - box.min_z)
    )
    accessible_volume_est = accessible_fraction * box_volume
    sphere_volume_est = box_volume * (1.0 - accessible_fraction)
    sphere_volume_exact = (4.0 / 3.0) * math.pi * (sphere.radius ** 3)

    return accessible_fraction, accessible_volume_est, sphere_volume_est, sphere_volume_exact


def accessible_dna_volume_from_walkers(walkers, dna_spheres, box, thin=1):
    """
    Estimate accessible fraction/volume given precomputed walkers.
    """
    accessible = 0
    total = 0

    x_in, y_in, z_in = [], [], []
    for w in walkers:
        for i in range(0, len(w['path_x']), thin):
            point = (w['path_x'][i], w['path_y'][i], w['path_z'][i])
            inside_any = False
            for (cx, cy, cz, r) in dna_spheres:
                if is_point_inside_sphere(point, Sphere(cx, cy, cz, r)):
                    inside_any = True
                    break
            if not inside_any:
                x_in.append(point[0])
                y_in.append(point[1])
                z_in.append(point[2])
                accessible += 1
            total += 1
    
    # Calculate volumes
    box_volume = (box.max_x - box.min_x) * (box.max_y - box.min_y) * (box.max_z - box.min_z)
    exact_volume = sum((4/3) * math.pi * (r**3) for (cx, cy, cz, r) in dna_spheres)
    accessible_fraction = (accessible / total) * 100

    accessible_volume = (accessible / total) * box_volume
    estimated_accessible_volume = box_volume - exact_volume

    # Plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x_in, y_in, z_in, c='b', marker='o', s=1, alpha=0.2)
    ax.set_title(f'Accessible Points in DNA Molecule')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

    return accessible_fraction, accessible_volume, exact_volume, estimated_accessible_volume


