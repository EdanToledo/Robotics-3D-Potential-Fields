import os
from torch import optim, nn, utils, Tensor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from main import check_for_collisions, check_for_completion, generate_all_positions

test = np.load("dataset.pl.npy", allow_pickle=True)


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = torch.tensor(self.data[idx][:-3]).float()

        y = torch.tensor(self.data[idx][-3:]).float()

        return x, y


class VectorFieldNeuralNetwork(pl.LightningModule):
    def __init__(self, num_spaceships=4):
        super().__init__()
        self.num_spaceships = num_spaceships
        self.other_goal_positions_encoder = nn.GRU(
            batch_first=True, input_size=1, hidden_size=8
        )
        self.other_spaceship_positions_encoder = nn.GRU(
            batch_first=True, input_size=1, hidden_size=8
        )
        self.obstacles_encoder = nn.GRU(batch_first=True, input_size=1, hidden_size=8)
        self.spaceship_position_encoder = nn.Linear(3, 8)
        self.goal_position_encoder = nn.Linear(3, 8)
        self.map_size_encoder = nn.Linear(1, 8)
        self.num_spaceships_encoder = nn.Linear(1, 8)

        self.final_layer = nn.Sequential(
            nn.Linear(8 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):

        map_embedding = self.map_size_encoder(x[:, 0].reshape(-1, 1))
        num_spaceships_embedding = self.num_spaceships_encoder(x[:, 1].reshape(-1, 1))
        spaceship_pos_embedding = self.spaceship_position_encoder(x[:, 2 : 2 + 3])
        goal_pos_embedding = self.goal_position_encoder(x[:, 5 : 5 + 3])
        other_spaceship_pos_embedding, _ = self.other_spaceship_positions_encoder(
            x[:, 8 : 8 + 3 * self.num_spaceships].reshape(x.shape[0], -1, 1)
        )
        other_spaceship_pos_embedding = other_spaceship_pos_embedding[:, -1, :]
        other_goal_pos_embedding, _ = self.other_goal_positions_encoder(
            x[:, 8 + 3 * self.num_spaceships : 8 + 6 * self.num_spaceships].reshape(
                x.shape[0], -1, 1
            )
        )
        other_goal_pos_embedding = other_goal_pos_embedding[:, -1, :]
        obstacles_embedding, _ = self.obstacles_encoder(
            x[:, 8 + 6 * self.num_spaceships :].reshape(x.shape[0], -1, 1)
        )
        obstacles_embedding = obstacles_embedding[:, -1, :]

        x = torch.concat(
            [
                map_embedding,
                num_spaceships_embedding,
                spaceship_pos_embedding,
                goal_pos_embedding,
                other_spaceship_pos_embedding,
                other_goal_pos_embedding,
                obstacles_embedding,
            ],-1
        )

        out = self.final_layer(x)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y = batch
        x = self.forward(x)
        loss = nn.functional.mse_loss(x, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer


def format_input_for_nn(
    map_size, spaceship_index, spaceships_positions, goal_positions, obstacle_positions
):
    
    data = []

    data.append(map_size)
    data.append(len(spaceships_positions))
    data.extend(spaceships_positions[spaceship_index].flatten())
    data.extend(goal_positions[spaceship_index].flatten())
    data.extend(spaceships_positions[np.arange(len(spaceships_positions))!=spaceship_index].flatten())
    data.extend(goal_positions[np.arange(len(goal_positions))!=spaceship_index].flatten())
    data.extend(obstacle_positions.flatten())

    return torch.tensor(data).float().reshape(1, -1)


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
    neural_net,
):

    for i in range(num_of_obstacles_meteorites):
        obstacle_positions[i] = obstacle_positions[i] + 0.2 * (
            np.array([np.cos(timestep), np.sin(timestep), np.cos(timestep)])
            * map_size
            / 2
        )

    num_spaceships = len(spaceships_positions)
    for spaceship_index in range(num_spaceships):
        position = spaceships_positions[spaceship_index]

        velocity = neural_net(
            format_input_for_nn(
                map_size,
                spaceship_index,
                spaceships_positions,
                goal_positions,
                obstacle_positions,
            )
        )
        spaceships_positions[spaceship_index] += torch.squeeze(velocity).detach().numpy()

        collided = check_for_collisions(
            spaceship_index,
            spaceships_positions,
            obstacle_positions,
            goal_positions,
            spaceship_radius + obstacle_radius + 0.01,
        )
        if collided:
            return collided

    return False


def eval(
    map_size,
    num_spaceships,
    goal_radius,
    num_obstacles,
    num_of_obstacles_meteorites,
    obstacle_radius,
    spaceship_radius,
    safety_distance,
    max_speed,
    attractive_force_scale,
    repulsive_force_scale,
    vortex_scale,
    timesteps,
    neural_net,
):
    """Evaluates a single episode. Returns True if it completed without any collisions. returns false if there were collisions or the spaceships didn't reach their goal."""

    spaceships_positions, goal_positions, obstacle_positions = generate_all_positions(
        num_spaceships,
        num_obstacles,
        map_size,
        spaceship_radius,
        goal_radius,
        obstacle_radius,
    )

    assert num_of_obstacles_meteorites <= num_obstacles
    with torch.no_grad():
        for i in range(timesteps):
            collision = perform_timestep(
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
                neural_net,
            )

            if collision:
                return False, i, False

            if check_for_completion(
                spaceships_positions,
                goal_positions,
                distance_limit=spaceship_radius + goal_radius,
            ):
                # print("All spaceships have made it to their goals...")
                return True, i, False

    # print("Not all spaceships have made it to their goals...")
    return False, timesteps, True


def experiment(
    map_size=5,
    num_spaceships=4,
    goal_radius=0.3,
    num_obstacles=3,
    num_of_obstacles_meteorites=0,
    obstacle_radius=0.3,
    spaceship_radius=0.2,
    safety_distance=1.0,
    max_speed=0.3,
    attractive_force_scale=0.3,
    repulsive_force_scale=0.3,
    vortex_scale=0.22,
    timesteps=150,
    episodes=100,
    neural_net=None,
):

    success = []
    failure_due_to_local_minima = []
    failure_due_to_collision = []
    episode_length = []
    for i in tqdm(range(episodes)):
        result, time, no_collision = eval(
            map_size,
            num_spaceships,
            goal_radius,
            num_obstacles,
            num_of_obstacles_meteorites,
            obstacle_radius,
            spaceship_radius,
            safety_distance,
            max_speed,
            attractive_force_scale,
            repulsive_force_scale,
            vortex_scale,
            timesteps,
            neural_net,
        )
        episode_length.append(time)
        if result:
            success.append(1)
        else:
            success.append(0)
            if no_collision:
                failure_due_to_local_minima.append(1)
                failure_due_to_collision.append(0)
            else:
                failure_due_to_local_minima.append(0)
                failure_due_to_collision.append(1)

    print(f"Average Completion Rate: {np.mean(success)}")
    print(f"Average Length of Episode: {np.mean(episode_length)}")
    print(f"Average Failure Due to Collision: {np.mean(failure_due_to_collision)}")
    print(
        f"Average Failure Due to Local Minima: {np.mean(failure_due_to_local_minima)}"
    )
    return (
        np.mean(success),
        np.mean(episode_length),
        np.mean(failure_due_to_collision),
        np.mean(failure_due_to_local_minima),
    )


def run_experiment(neural_net):
    success_rates = []
    lengths = []
    collision_failures = []
    local_minimas = []

    success_rate, length, collision_failure, local_minima = experiment(
        map_size=10,
        timesteps=500,
        episodes=1000,
        spaceship_radius=0.1,
        safety_distance=0.3,
        vortex_scale=0.25,
        num_obstacles=5,
        num_of_obstacles_meteorites=2,
        num_spaceships=6,
        neural_net=neural_net,
    )
    success_rates.append(success_rate)
    lengths.append(length)
    collision_failures.append(collision_failure)
    local_minimas.append(local_minima)

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(success_rates)
    axes[0, 0].set_title("Average percentage of successfully completed runs")
    axes[0, 1].plot(lengths)
    axes[0, 1].set_title("Average length of runs")
    axes[1, 0].plot(collision_failures)
    axes[1, 0].set_title("Average percentage of failures due to collisions")
    axes[1, 1].plot(local_minimas)
    axes[1, 1].set_title("Average percentage of failures due to local minimas")
    plt.show()


def main():

    vf_nn = VectorFieldNeuralNetwork()

    # setup data
    # dataset = CustomDataset(test)
    # train_loader = DataLoader(
    #     dataset, batch_size=128, shuffle=True, num_workers=4, prefetch_factor=2
    # )

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # trainer = pl.Trainer(limit_train_batches=1.0, max_epochs=10)
    # trainer.fit(model=vf_nn, train_dataloaders=train_loader)

    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=9-step=175210.ckpt"
    vf_nn = VectorFieldNeuralNetwork.load_from_checkpoint(checkpoint, num_spaceships=6)

    # # choose your trained nn.Module
    neural_net = vf_nn
    neural_net.eval()

    run_experiment(neural_net)


if __name__ == "__main__":
    main()
