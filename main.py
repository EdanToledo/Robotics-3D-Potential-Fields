import copy
import os
import numpy as np
import matplotlib.pylab as plt
import imageio.v2 as imageio
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm


def generate_random_positions(
    num_positions: int, object_radius: float, map_size: float, num_coordinates: int = 3
) -> np.ndarray:
    """Generate three random positions representing x, y, z coordinates.

    Args:
        num_positions (int): The number of positions to generate.
        object_radius (float): The radius of the object the position is being generated for. This is to keep the position within the world boundaries.
        map_size (float): The size of the map. This sets the boundaries to be between (-map_size, map_size)
        num_coordinates (int): The number of coordinates. Defaults to 3.

    Returns:
        np.ndarray: The generated positions in the shape (num_positions, 3)
    """

    lower_bound = -map_size + object_radius
    upper_bound = map_size - object_radius

    positions = np.random.uniform(
        lower_bound, upper_bound, size=(num_positions, num_coordinates)
    )

    return positions


def cap(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    """Caps a velocity to be below max speed.

    Args:
        velocity (np.ndarray): The velocity being checked.
        max_speed (float): The max speed to cap the velocity with.

    Returns:
        np.ndarray: The capped velocity.
    """
    n = np.linalg.norm(velocity, axis=-1)
    if n > max_speed:
        return (velocity / n) * max_speed
    return velocity


def add_noise_to_velocity(
    velocity: np.ndarray, epsilon: float = 0.01, noise_scale: float = 0.2
) -> np.ndarray:
    """Add noise to velocity that is below epsilon.

    Args:
        velocity (np.ndarray): The velocity that noise is potentially being added to.
        epsilon (float, optional): The threshold whereby any velocity that has a magnitude lower than it will have noise added. Defaults to 0.01.
        noise_scale (float, optional): The standard deviation of the normal distribution used to generate the noise. Defaults to 0.2.

    Returns:
        np.ndarray: The new velocity.
    """

    if np.linalg.norm(velocity, axis=-1) < epsilon:
        noise = np.random.normal(loc=0, scale=noise_scale, size=2)
        velocity += noise

    return velocity


def get_vortex_velocity(
    position: np.ndarray,
    centre_pos: np.ndarray,
    centre_radius: float,
    max_speed: float,
    scale: float = 0.07,
    safety_distance: float = 0.5,
) -> np.ndarray:
    """Get velocity from vortex.

    Args:
        position (np.ndarray): The position to evaluate what the velocity is.
        centre_pos (np.ndarray): The centre of the vortex.
        max_speed (float): The max speed of the velocity.
        scale (float, optional): The value by which to scale the velocity. Defaults to 0.07.
        safety_distance (float, optional): The distance around the centre that the vortex vectors actually effect. Defaults to .5.

    Returns:
        np.ndarray: _description_
    """

    velocity = np.zeros(3, dtype=np.float32)

    # 2D
    # v_pos = position - centre_pos
    # distance = np.linalg.norm(v_pos)
    # square_sum = (v_pos[0] ** 2) + (v_pos[1] ** 2)

    # if distance < centre_radius + safety_distance:
    #     velocity[0] = (-v_pos[1] / square_sum) / distance
    #     velocity[1] = (v_pos[0] / square_sum) / distance

    # velocity *= scale

    # 3D
    v_pos = position - centre_pos
    distance = np.linalg.norm(v_pos)
    square_sum = (v_pos[0] ** 2) + (v_pos[1] ** 2)

    if distance < centre_radius + safety_distance:
        velocity[0] = (-v_pos[1] / square_sum) / distance
        velocity[1] = (v_pos[0] / square_sum) / distance

    velocity *= scale
    velocity = cap(velocity, max_speed)

    return velocity


def get_repulsive_velocity(
    position: np.ndarray,
    centre_pos: np.ndarray,
    centre_radius: float,
    repel_distance_limit: float,
    scale: float,
    max_speed: float,
) -> np.ndarray:

    velocity = np.zeros(3, dtype=np.float32)

    # 2D
    # x_dist = centre_pos[0] - position[0]
    # y_dist = centre_pos[1] - position[1]
    # distance = np.sqrt((x_dist**2) + (y_dist**2))
    # theta = np.arctan2(y_dist, x_dist)

    # if distance < centre_radius + repel_distance_limit:
    #     velocity[0] = -(scale) * (np.cos(theta))
    #     velocity[1] = -(scale) * (np.sin(theta))

    # 3D
    dist = position - centre_pos
    limit = repel_distance_limit + centre_radius
    if np.linalg.norm(dist) < limit:
        velocity = scale * ((1 / dist) - (1 / limit))

    velocity = cap(velocity, max_speed)

    return velocity


def get_attractive_velocity(
    position: np.ndarray,
    centre_pos: np.ndarray,
    scale: float,
    max_speed: float,
    goal_radius: float,
    safety_distance: float,
) -> np.ndarray:
    v = np.zeros(2, dtype=np.float32)

    # 2D
    # x_dist = centre_pos[0] - position[0]
    # y_dist = centre_pos[1] - position[1]
    # distance = np.sqrt((x_dist**2) + (y_dist**2))
    # theta = np.arctan2(y_dist, x_dist)

    # v[0] = scale * max(distance - goal_radius, 0) * np.cos(theta)
    # v[1] = scale * max(distance - goal_radius, 0) * np.sin(theta)

    # if distance > goal_radius and distance < goal_radius + safety_distance:
    #     v[0] = scale * (distance - goal_radius) * np.cos(theta)
    #     v[1] = scale * (distance - goal_radius) * np.sin(theta)
    # elif distance > safety_distance + goal_radius:
    #     v[0] = scale * safety_distance * np.cos(theta)
    #     v[1] = scale * safety_distance * np.sin(theta)

    # 3D
    
    v = -scale * (position - centre_pos)

    v = cap(v, max_speed)

    return v


def get_velocity(
    position,
    goal_positions,
    goal_radius,
    spaceship_index,
    spaceships_positions,
    spaceship_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
):

    other_spaceships_velocity = np.zeros(3)
    other_goals_velocity = np.zeros(3)
    obstacles_velocity = np.zeros(3)
    attractive_velocity = np.zeros(3)

    goal_position = goal_positions[spaceship_index]

    attractive_velocity = get_attractive_velocity(
        position,
        goal_position,
        attractive_force_scale,
        max_speed,
        goal_radius,
        safety_distance,
    )

    for i, spaceship_pos in enumerate(spaceships_positions):
        if i == spaceship_index:
            continue

        other_spaceships_velocity += get_repulsive_velocity(
            position,
            spaceship_pos,
            spaceship_radius,
            safety_distance,
            repulsive_force_scale,
            max_speed,
            )
        
        other_spaceships_velocity +=get_vortex_velocity(
            position,
            spaceship_pos,
            spaceship_radius,
            max_speed,
            vortex_scale,
            safety_distance,
        )

    for i, other_goal_pos in enumerate(goal_positions):
        if i == spaceship_index:
            continue

        other_goals_velocity += get_repulsive_velocity(
            position,
            other_goal_pos,
            goal_radius,
            safety_distance,
            repulsive_force_scale,
            max_speed,
        ) 
        other_goals_velocity += get_vortex_velocity(
            position,
            other_goal_pos,
            goal_radius,
            max_speed,
            vortex_scale,
            safety_distance,
        )

    for i, obstacle_pos in enumerate(obstacle_positions):

        obstacles_velocity += get_repulsive_velocity(
            position,
            obstacle_pos,
            obstacle_radius,
            safety_distance,
            repulsive_force_scale,
            max_speed,
        )
        obstacles_velocity += get_vortex_velocity(
            position,
            obstacle_pos,
            obstacle_radius,
            max_speed,
            vortex_scale,
            safety_distance,
        )

    velocity = (
        attractive_velocity
        + other_spaceships_velocity
        + obstacles_velocity
        + other_goals_velocity
    )
    velocity = cap(velocity, max_speed)

    return velocity


def check_for_collisions(
    spaceship_index,
    spaceships_positions,
    obstacle_positions,
    goal_positions,
    distance_limit,
):
    current_spaceship_pos = spaceships_positions[spaceship_index]
    other_spaceship_positions = np.delete(spaceships_positions, spaceship_index, axis=0)
    other_goals = np.delete(goal_positions, spaceship_index, axis=0)

    all_positions_to_check = np.concatenate(
        [other_spaceship_positions, other_goals, obstacle_positions], axis=0
    )
    for other_pos in all_positions_to_check:
        if np.linalg.norm(current_spaceship_pos - other_pos, axis=-1) < distance_limit:
            print("Collision Detected!")
            return True


    return False

def check_for_completion(
    spaceships_positions,
    goal_positions,
    distance_limit,
):
   
    for i, spaceship_pos in enumerate(spaceships_positions):
        if np.linalg.norm(spaceship_pos - goal_positions[i], axis=-1) > distance_limit:
            return False

    return True


def perform_timestep(
    spaceships_positions,
    spaceship_radius,
    goal_positions,
    goal_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
    num_of_obstacles_meteorites,
    map_size,
    timestep,
):
    # TODO Add different movement for meteor
    for i in range(num_of_obstacles_meteorites):
        obstacle_positions[i] = (
                obstacle_positions[i]
                + 0.2*(np.array([np.cos(timestep), np.sin(timestep), np.cos(timestep)])
                * map_size/2) 
            )

    num_spaceships = len(spaceships_positions)
    for spaceship_index in range(num_spaceships):
        position = spaceships_positions[spaceship_index]

        velocity = get_velocity(
            position,
            goal_positions,
            goal_radius,
            spaceship_index,
            spaceships_positions,
            spaceship_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
        )
        spaceships_positions[spaceship_index] += velocity

        check_for_collisions(
            spaceship_index,
            spaceships_positions,
            obstacle_positions,
            goal_positions,
            spaceship_radius + obstacle_radius,
        )

    return spaceships_positions, obstacle_positions


def plot_2dvf(
    map_size,
    spaceship_index,
    spaceships_positions,
    spaceship_radius,
    goal_positions,
    goal_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
    title="",
    axis=None,
    axis_width=2,
):
    row = int(spaceship_index * 2 // axis_width)
    column = int((spaceship_index * 2) % axis_width)

    num_points = 20

    X, Y, Z = np.meshgrid(
        np.linspace(-map_size, map_size, num_points),
        np.linspace(-map_size, map_size, num_points),
        np.linspace(-map_size, map_size, num_points),
    )
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    W = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            for k in range(len(X[0])):
                velocity = get_velocity(
                    np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]),
                    goal_positions,
                    goal_radius,
                    spaceship_index,
                    spaceships_positions,
                    spaceship_radius,
                    obstacle_positions,
                    obstacle_radius,
                    safety_distance,
                    attractive_force_scale,
                    repulsive_force_scale,
                    vortex_scale,
                    max_speed,
                )
                U[i, j, k] = velocity[0]
                V[i, j, k] = velocity[1]
                W[i, j, k] = velocity[2]

    # Plot XY Plane
    axis[row][column].quiver(
        X[:, :, 0], Y[:, :, 0], U[:, :, 0], V[:, :, 0], units="width"
    )
    # Plot XZ Plane
    axis[row][column + 1].quiver(
        X[0, :, :], Z[:, 0, :], U[0, :, :], W[:, 0, :], units="width"
    )

    # XY Plane
    axis[row, column] = create_circle(
        spaceships_positions[spaceship_index],
        spaceship_radius,
        axis[row, column],
        "cyan",
    )

    # # XZ Plane
    axis[row, column + 1] = create_circle(
        spaceships_positions[spaceship_index],
        spaceship_radius,
        axis[row, column + 1],
        "cyan",
        xy=False,
    )

    # # XY Plane
    axis[row, column] = create_circle(
        goal_positions[spaceship_index], goal_radius, axis[row][column], "magenta"
    )
    # # XZ Plane
    axis[row, column + 1] = create_circle(
        goal_positions[spaceship_index],
        goal_radius,
        axis[row][column + 1],
        "magenta",
        False,
    )

    for i, spaceship in enumerate(spaceships_positions):
        if i == spaceship_index:
            continue
        # XY
        axis[row][column] = create_circle(
            spaceship, spaceship_radius, axis[row][column], "red"
        )
        # XZ
        axis[row][column + 1] = create_circle(
            spaceship, spaceship_radius, axis[row][column + 1], "red", False
        )

    for i, spaceship in enumerate(goal_positions):
        if i == spaceship_index:
            continue
        # XY
        axis[row][column] = create_circle(
            goal_positions[i], goal_radius, axis[row][column], "firebrick"
        )
        # XZ
        axis[row][column + 1] = create_circle(
            goal_positions[i], goal_radius, axis[row][column + 1], "firebrick", False
        )

    for i, obstacle in enumerate(obstacle_positions):
        # XY
        axis[row][column] = create_circle(
            obstacle, obstacle_radius, axis[row][column], "black"
        )
        # XZ
        axis[row][column + 1] = create_circle(
            obstacle, obstacle_radius, axis[row][column + 1], "black", False
        )

    axis[row][column].set_title(f"Spaceship {spaceship_index} XY Vector Field " + title)
    axis[row][column + 1].set_title(
        f"Spaceship {spaceship_index} XZ Vector Field " + title
    )
    return axis


def plot_all_2dvfs(
    map_size,
    num_spaceships,
    spaceships_positions,
    spaceship_radius,
    goal_positions,
    goal_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
):

    if num_spaceships % 2 == 0:
        row, col = num_spaceships // 2, num_spaceships * 2 // 2
    else:
        row, col = (num_spaceships // 2) + 1, num_spaceships * 2 // 2

    figure, axis = plt.subplots(row, col, squeeze=False, figsize=(18, 18))

    for spaceship_index in range(num_spaceships):

        axis = plot_2dvf(
            map_size,
            spaceship_index,
            spaceships_positions,
            spaceship_radius,
            goal_positions,
            goal_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
            "",
            axis,
            col,
        )

    if num_spaceships % 2 != 0:
        figure.delaxes(axis[-1, -1])

    return figure, axis


def create_circle(position, radius, axis, colour, xy=True):
    if xy:
        axis.add_artist(
            plt.Circle(np.array([position[0], position[1]]), radius, color=colour)
        )
    else:
        axis.add_artist(
            plt.Circle(np.array([position[0], position[-1]]), radius, color=colour)
        )
    return axis


def plot_3dvf(
    map_size,
    spaceship_index,
    spaceships_positions,
    spaceship_radius,
    goal_positions,
    goal_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
    title="",
    axis=None,
    axis_width=2,
):
    row = int(spaceship_index // axis_width)
    column = int(spaceship_index % axis_width)

    num_points = int(map_size*4)

    X, Y, Z = np.meshgrid(
        np.linspace(-map_size, map_size, num_points),
        np.linspace(-map_size, map_size, num_points),
        np.linspace(-map_size, map_size, num_points),
    )
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    W = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            for k in range(len(X[0])):
                velocity = get_velocity(
                    np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]),
                    goal_positions,
                    goal_radius,
                    spaceship_index,
                    spaceships_positions,
                    spaceship_radius,
                    obstacle_positions,
                    obstacle_radius,
                    safety_distance,
                    attractive_force_scale,
                    repulsive_force_scale,
                    vortex_scale,
                    max_speed,
                )
                U[i, j, k] = velocity[0]
                V[i, j, k] = velocity[1]
                W[i, j, k] = velocity[2]

    c = np.arctan2(V, U)
    # Flatten and normalize
    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.hsv(c)

    axis[row][column].quiver(X, Y, Z, U, V, W, normalize=True, alpha=0.7, colors = c)

    axis[row][column] = create_sphere(
        spaceships_positions[spaceship_index],
        spaceship_radius,
        axis[row][column],
        "cyan",
    )

    axis[row][column] = create_sphere(
        goal_positions[spaceship_index], goal_radius, axis[row][column], "magenta"
    )

    for i, spaceship in enumerate(spaceships_positions):
        if i == spaceship_index:
            continue

        axis[row][column] = create_sphere(
            spaceship, spaceship_radius, axis[row][column], "red"
        )

    for i, spaceship in enumerate(goal_positions):
        if i == spaceship_index:
            continue
        axis[row][column] = create_sphere(
            goal_positions[i], goal_radius, axis[row][column], "firebrick"
        )

    for i, obstacle in enumerate(obstacle_positions):

        axis[row][column] = create_sphere(
            obstacle, obstacle_radius, axis[row][column], "black"
        )

    axis[row][column].set_title(f"Spaceship {spaceship_index} Vector Field " + title)
    return axis


def plot_all_3dvfs(
    map_size,
    num_spaceships,
    spaceships_positions,
    spaceship_radius,
    goal_positions,
    goal_radius,
    obstacle_positions,
    obstacle_radius,
    safety_distance,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    max_speed,
):

    if num_spaceships % 2 == 0:
        row, col = num_spaceships // 2, num_spaceships // 2
    else:
        row, col = (num_spaceships // 2) + 1, num_spaceships // 2

    axis = []
    index = 1
    figure = plt.figure(figsize=(50, 50))
    for i in range(row):
        axis.append([])
        for j in range(col):
            ax = figure.add_subplot(int(f"{row}{col}{index}"), projection="3d")
            axis[i].append(ax)
            index += 1

    for spaceship_index in range(num_spaceships):

        axis = plot_3dvf(
            map_size,
            spaceship_index,
            spaceships_positions,
            spaceship_radius,
            goal_positions,
            goal_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
            "",
            axis,
            col,
        )

    if num_spaceships % 2 != 0:
        figure.delaxes(axis[-1, -1])

    return figure, axis


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def delete_files(file_dir):
    files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
    for filename in set(files):
        os.remove(f"{file_dir}/{filename}")


def build_video(video_name, file_dir, gif=False, fps=20):
    files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
    files.sort(key=natural_keys)

    with imageio.get_writer(
        f'{video_name}.{"gif" if gif else "mp4"}', mode="I", fps=fps
    ) as writer:
        for filename in files:
            if filename[0] == ".":
                continue
            image = imageio.imread(f"{file_dir}/{filename}")
            writer.append_data(image)

    delete_files(file_dir)


def create_sphere(position, sphere_radius, ax, colour):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = position[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    y = position[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    z = position[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=colour)
    return ax


def main(
    map_size=6,
    num_spaceships=4,
    goal_radius=0.3,
    num_obstacles=1,
    num_of_obstacles_meteorites=0,
    obstacle_radius=0.3,
    spaceship_radius=0.2,
    safety_distance=1.0,
    max_speed=0.3,
    attractive_force_scale=0.3,
    repulsive_force_scale=0.3,
    vortex_scale=0.22,
    timesteps=0,
    plots_folder="./plots",
    three_dimensional_plotting=True,
    gif = True
):

    spaceships_positions = generate_random_positions(
        num_spaceships, spaceship_radius, map_size
    )
    goal_positions = generate_random_positions(
        num_spaceships, spaceship_radius, map_size
    )
    obstacle_positions = generate_random_positions(
        num_obstacles, obstacle_radius, map_size
    )

    assert num_of_obstacles_meteorites < num_obstacles
    if three_dimensional_plotting:
        plot_all_3dvfs(
            map_size,
            num_spaceships,
            spaceships_positions,
            spaceship_radius,
            goal_positions,
            goal_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
        )
    else:
        plot_all_2dvfs(
            map_size,
            num_spaceships,
            spaceships_positions,
            spaceship_radius,
            goal_positions,
            goal_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
        )

    plt.savefig(f"{plots_folder}/1.png")

    plt.close()
    for i in tqdm(range(2, timesteps + 2)):
        perform_timestep(
            spaceships_positions,
            spaceship_radius,
            goal_positions,
            goal_radius,
            obstacle_positions,
            obstacle_radius,
            safety_distance,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            max_speed,
            num_of_obstacles_meteorites,
            map_size,
            i,
        )
        if three_dimensional_plotting:
            plot_all_3dvfs(
                map_size,
                num_spaceships,
                spaceships_positions,
                spaceship_radius,
                goal_positions,
                goal_radius,
                obstacle_positions,
                obstacle_radius,
                safety_distance,
                attractive_force_scale,
                repulsive_force_scale,
                vortex_scale,
                max_speed,
            )
        else:
            plot_all_2dvfs(
                map_size,
                num_spaceships,
                spaceships_positions,
                spaceship_radius,
                goal_positions,
                goal_radius,
                obstacle_positions,
                obstacle_radius,
                safety_distance,
                attractive_force_scale,
                repulsive_force_scale,
                vortex_scale,
                max_speed,
            )
        plt.savefig(f"{plots_folder}/{i}.png")

        plt.close()

        if check_for_completion(spaceships_positions, goal_positions, distance_limit=spaceship_radius+goal_radius):
            print("All Spaceships have made it to their goals...")
            break

    print("Building Video...")
    build_video("anim", plots_folder, gif=gif, fps=5)


if __name__ == "__main__":
    main()
