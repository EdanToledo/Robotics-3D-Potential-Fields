import os
from torch import optim, nn, utils, Tensor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from main import check_for_collisions, check_for_completion, generate_all_positions




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
    def __init__(self):
        super().__init__()
       
        embedding_dim = 32
        # self.map_size_encoder = nn.Linear(1, embedding_dim)
        # self.num_spaceships_encoder = nn.Linear(1, embedding_dim)
        self.spaceship_position_encoder = nn.Linear(3, embedding_dim)
        self.goal_position_encoder = nn.Linear(3, embedding_dim)
        
        self.obstacles_encoder = nn.GRU(batch_first=True, input_size=3, hidden_size=embedding_dim*2, num_layers=1)

        self.final_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(4*embedding_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        )

        self.loss_fn = nn.HuberLoss()
        self.train_loader, self.valid_loader = create_all_dataloaders("dataset",128)

    def forward(self, x):
        
        # map_size = x[:, 0].reshape(x.shape[0], 1)
        # num_spaceships = x[:, 1].reshape(x.shape[0], 1)
        current_spaceship_pos = x[:, 2 : 5]
        current_goal_pos = x[:, 5 : 8]
        # map_embedding = self.map_size_encoder(map_size)
        # num_spaceships_embedding = self.num_spaceships_encoder(num_spaceships)
        spaceship_pos_embedding = self.spaceship_position_encoder(current_spaceship_pos)
        goal_pos_embedding = self.goal_position_encoder(current_goal_pos)

        sequence_of_obstacles = x[:, 8 : ].reshape(x.shape[0],-1, 3)
        
        all_obstacles_embedding, _= self.obstacles_encoder(sequence_of_obstacles)
        all_obstacles_embedding = all_obstacles_embedding[:, -1, :]
        

        x = torch.concat(
            [
                # map_embedding,
                # num_spaceships_embedding,
                spaceship_pos_embedding,
                goal_pos_embedding,
                all_obstacles_embedding
            ], dim = -1
        )

        out = self.final_layer(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss) 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

   

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


        velocity = neural_net(
            format_input_for_nn(
                map_size,
                spaceship_index,
                spaceships_positions,
                goal_positions,
                obstacle_positions,
            )
        )
        
        spaceships_positions[spaceship_index] += torch.squeeze(velocity).numpy()

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
    map_size=10,
    num_spaceships=4,
    goal_radius=0.3,
    num_obstacles=5,
    num_of_obstacles_meteorites=2,
    obstacle_radius=0.3,
    spaceship_radius=0.2,
    safety_distance=1.0,
    max_speed=0.3,
    attractive_force_scale=0.3,
    repulsive_force_scale=0.3,
    vortex_scale=0.25,
    timesteps=500,
    episodes=1000,
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


def run_experiment(neural_net, min_range, max_range, variable, optional_set=None):
    success_rates = []
    lengths = []
    collision_failures = []
    local_minimas = []
    params = {"map_size":10,
            "timesteps":500,
            "episodes":1000,
            "spaceship_radius":0.1,
            "safety_distance":0.3,
            "vortex_scale":0.25,
            "num_obstacles":5,
            "num_of_obstacles_meteorites":2,
            "num_spaceships":4,
            "neural_net" : neural_net}

    if optional_set is not None:
        values = optional_set
    else:
        values = range(min_range, max_range)
    for i in values:
        params[variable] = i
        params["num_of_obstacles_meteorites"] = i-2
        success_rate, length, collision_failure, local_minima = experiment(
            **params
        )
        success_rates.append(success_rate)
        lengths.append(length)
        collision_failures.append(collision_failure)
        local_minimas.append(local_minima)

    fig, axes = plt.subplots(2, 2)

    print("Success Rates:",success_rates)
    print("Lengths:",lengths)
    print("Collision Failures:",collision_failures)
    print("Local Minimas:",local_minimas)

    axes[0, 0].plot(success_rates)
    axes[0, 0].set_title("Average percentage of successfully completed runs")
    axes[0, 1].plot(lengths)
    axes[0, 1].set_title("Average length of runs")
    axes[1, 0].plot(collision_failures)
    axes[1, 0].set_title("Average percentage of failures due to collisions")
    axes[1, 1].plot(local_minimas)
    axes[1, 1].set_title("Average percentage of failures due to local minimas")
    plt.show()

def collate_batch(batch):
    """For padding"""
   
    xs, ys, = [], []
   
    for (x,y) in batch:
        ys.append(y)
        xs.append(x)
   
    ys = torch.stack(ys)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
   
    return xs, ys

def create_all_dataloaders(dataset_name, batch_size, train_test_split = 0.9, num_workers = 4):
    
    data = np.load(f"{dataset_name}.npy", allow_pickle=True)
    dataset = CustomDataset(data)
    train_set_size = int(train_test_split*len(dataset))
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
    train_loader = DataLoader(train_set, batch_size, True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size, False, num_workers=num_workers)
  
    return train_loader, valid_loader

def main():
    load_check = 1
    train = 0

    
    if load_check:
        version = 16
        epoch = 49
        step = 798200
        checkpoint = f"./lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt"
        vf_nn = VectorFieldNeuralNetwork.load_from_checkpoint(checkpoint)
    else:
        vf_nn = VectorFieldNeuralNetwork()

    # setup data
    if train:
        
        trainer = pl.Trainer(limit_train_batches=1.0, max_epochs=50)
        trainer.fit(model=vf_nn)
    else:
       
        # # choose your trained nn.Module
        neural_net = vf_nn
        neural_net.eval()

        run_experiment(neural_net, 2, 12, "num_obstacles")


if __name__ == "__main__":
    main()
