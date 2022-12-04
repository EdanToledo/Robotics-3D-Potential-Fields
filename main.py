import os
import numpy as np
import matplotlib.pylab as plt
import imageio.v2 as imageio
from os import listdir
from os.path import isfile, join
import re

def generate_random_positions(
    num_positions: int, object_radius: float, map_size: float, num_coordinates: int = 2
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
    n = np.linalg.norm(velocity)
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
    position: np.ndarray, centre_pos: np.ndarray, max_speed: float, scale: float = 0.07, safety_distance : float = .5
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

    velocity = np.zeros(2, dtype=np.float32)

    v_pos = position - centre_pos
    distance = np.linalg.norm(v_pos)
    square_sum = (v_pos[0] ** 2) + (v_pos[1] ** 2)

    if distance > safety_distance:
        velocity[0] = (-position[1] / square_sum) / distance
        velocity[1] = (position[0] / square_sum) / distance 
   
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

    velocity = np.zeros(2, dtype=np.float32)

    x_dist = centre_pos[0] - position[0]
    y_dist = centre_pos[1] - position[1]
    distance = np.sqrt((x_dist**2) + (y_dist**2))
    theta = np.arctan2(y_dist, x_dist)

    if distance < centre_radius + repel_distance_limit:
        velocity[0] = -(scale) * (np.cos(theta))
        velocity[1] = -(scale) * (np.sin(theta))

    velocity = cap(velocity, max_speed)

    return velocity


def get_attractive_velocity(
    position: np.ndarray, centre_pos: np.ndarray, scale: float, max_speed: float, goal_radius: float, safety_distance : float
) -> np.ndarray:
    v = np.zeros(2, dtype=np.float32)

    x_dist = centre_pos[0] - position[0]
    y_dist = centre_pos[1] - position[1]
    distance = np.sqrt((x_dist**2) + (y_dist**2))
    theta = np.arctan2(y_dist, x_dist)
    
    v[0] = scale * max(distance - goal_radius, 0) * np.cos(theta)
    v[1] = scale * max(distance - goal_radius, 0) * np.sin(theta)

    if distance > goal_radius and distance < goal_radius + safety_distance:
        v[0] = scale * (distance - goal_radius) * np.cos(theta)
        v[1] = scale * (distance - goal_radius) * np.sin(theta)
    elif distance > safety_distance + goal_radius:
        v[0] = scale * safety_distance * np.cos(theta)
        v[1] = scale * safety_distance * np.sin(theta)

    v = cap(v, max_speed)

    return v


def get_velocity(
    position,
    goal_position,
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
    velocity = get_attractive_velocity(
        position, goal_position, attractive_force_scale, max_speed, goal_radius, safety_distance
    )

    other_spaceships_velocity = np.zeros(2)
    obstacles_velocity = np.zeros(2)

    for i, spaceship_pos in enumerate(spaceships_positions):
        if i == spaceship_index:
            continue
        spaceship_avoid_velocity = get_repulsive_velocity(
            position,
            spaceship_pos,
            spaceship_radius,
            safety_distance,
            repulsive_force_scale,
            max_speed,
        )
        spaceship_vortex_velocity = get_vortex_velocity(
            position, spaceship_pos, max_speed, vortex_scale, safety_distance
        )
        other_spaceships_velocity += (
            spaceship_avoid_velocity + spaceship_vortex_velocity
        )

    for i, obstacle_pos in enumerate(obstacle_positions):

        obstacle_avoid_velocity = get_repulsive_velocity(
            position,
            obstacle_pos,
            obstacle_radius,
            safety_distance,
            repulsive_force_scale,
            max_speed,
        )
        obstacle_vortex_velocity = get_vortex_velocity(
            position, obstacle_pos, max_speed, vortex_scale, safety_distance
        )
        obstacles_velocity += obstacle_avoid_velocity + obstacle_vortex_velocity

    velocity = velocity + other_spaceships_velocity+ obstacles_velocity
    velocity = cap(velocity, max_speed)

    return velocity


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
):
    num_spaceships = len(spaceships_positions)
    for spaceship_index in range(num_spaceships):
        position = spaceships_positions[spaceship_index]
        goal_position = goal_positions[spaceship_index]
        velocity = get_velocity(
            position,
            goal_position,
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

    return spaceships_positions


def plot_vf(
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
    shape_size = 200
):
    row = int(spaceship_index // axis_width)
    column = int(spaceship_index % axis_width)

    goal_pos = goal_positions[spaceship_index]
    X, Y = np.meshgrid(
        np.linspace(-map_size, map_size, 30), np.linspace(-map_size, map_size, 30)
    )
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            velocity = get_velocity(
                np.array([X[i, j], Y[i, j]]),
                goal_pos,
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
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
    axis[row, column].quiver(X, Y, U, V, units="width")
   
    axis[row, column].scatter(
        np.array([spaceships_positions[spaceship_index][0]]),
        np.array([spaceships_positions[spaceship_index][1]]),
        marker="o",
        s=shape_size,
        c="blue",
        label=f"Spaceship {spaceship_index} Position",
    )
    axis[row, column].scatter(
        np.array([goal_pos[0]]),
        np.array([goal_pos[1]]),
        marker="*",
        s=shape_size,
        c="blue",
        label=f"Spaceship {spaceship_index} Goal",
    )
    for i, spaceship in enumerate(spaceships_positions):
        if i == spaceship_index:
            continue
        axis[row, column].scatter(
            np.array([spaceship[0]]),
            np.array([spaceship[1]]),
            marker="o",
            s=shape_size,
            c="red",
            label=f"Spaceship {i} Position",
        )

    for i, obstacle in enumerate(obstacle_positions):
        
        axis[row, column].scatter(
            np.array([obstacle[0]]),
            np.array([obstacle[1]]),
            marker="o",
            s=shape_size,
            c="gray",
            label=f"Obstacle {i} Position",
        )

    axis[row, column].set_title(f"Spaceship {spaceship_index} Vector Field " + title)
    return axis

def plot_all_vfs(map_size,
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
            max_speed):
    
    if num_spaceships % 2 == 0:
        row, col = num_spaceships // 2, num_spaceships // 2
    else:
        row, col = (num_spaceships // 2) + 1, num_spaceships // 2

    figure, axis = plt.subplots(row, col, squeeze=False, figsize=(15, 15))

    for spaceship_index in range(num_spaceships):
        axis = plot_vf(
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
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def delete_files(file_dir):
    files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
    for filename in set(files):
        os.remove(f"{file_dir}/{filename}")

def build_video(video_name, file_dir,  gif = False, fps = 20):
    files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
    files.sort(key=natural_keys)
    
    with imageio.get_writer(f'{video_name}.{"gif" if gif else "mp4"}', mode='I',fps=fps) as writer:
        for filename in files:
            if filename[0] == ".":
                continue
            image = imageio.imread(f"{file_dir}/{filename}")
            writer.append_data(image)

    delete_files(file_dir)


# TODO
# def sphere():
#     r = 0.05
#     u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
#     x = np.cos(u) * np.sin(v)
#     y = np.sin(u) * np.sin(v)
#     z = np.cos(v)
#     ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

def main():
    map_size = 12
    num_spaceships = 4
    goal_radius = 0.5
    num_obstacles = 3
    obstacle_radius = 1.
    spaceship_radius = 0.5
    safety_distance = 3.
    max_speed = 0.25
    attractive_force_scale = max_speed
    repulsive_force_scale = max_speed
    vortex_scale = 0.2
    timesteps = 150
    plots_folder = "./plots"
    spaceships_positions = generate_random_positions(
        num_spaceships, spaceship_radius, map_size, 2
    )
    goal_positions = generate_random_positions(
        num_spaceships, spaceship_radius, map_size, 2
    )
    obstacle_positions = generate_random_positions(
        num_obstacles, obstacle_radius, map_size, 2
    )
  

    plot_all_vfs(map_size,
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
            max_speed)
   
    plt.savefig(f'{plots_folder}/1.png')
    
    plt.close()
    for i in range(2,timesteps+2):
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
        )

        plot_all_vfs(map_size,
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
                max_speed)
        plt.savefig(f'{plots_folder}/{i}.png')
       
        plt.close()
  

    build_video("anim",plots_folder,gif=False, fps = 10)
    

if __name__ == "__main__":
    main()