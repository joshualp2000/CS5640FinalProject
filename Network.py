import os

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_size, i, save_dir):
        super(Network, self).__init__()

        self.i = i
        self.save_dir = save_dir

        self.critic_fc1 = nn.Linear(23, 64)
        self.critic = nn.Linear(64, 1)

        self.actor_fc1 = nn.Linear(23, 64)
        self.actor_fc2 = nn.Linear(64, 64)

        self.movement = nn.Linear(64, 9)
        self.turning = nn.Linear(64, 9)
        self.jumping = nn.Linear(64, 9)

        self.float()

    def forward(self, x):
        critic = self.critic_fc1(x)
        critic = T.tanh(critic)
        critic = self.critic(critic)

        actor = self.actor_fc1(x)
        actor = T.tanh(actor)
        actor = self.actor_fc2(actor)
        actor = T.tanh(actor)

        movement = self.movement(actor)
        turning = self.turning(actor)
        jumping = self.jumping(actor)

        return critic, T.softmax(movement, dim=0), T.softmax(turning, dim=0), T.softmax(jumping, dim=0)

    def save_checkpoint(self, subfolder, epoch):
        location = self.save_dir + "/weights" + subfolder + "/agent_" + str(self.i) + "_iter_" + str(epoch)
        T.save(self.state_dict(), location)

    def load_checkpoint(self, subfolder, epoch):
        location = self.save_dir + "/weights" + subfolder + "/agent_" + str(self.i) + "_iter_" + str(epoch)
        self.load_state_dict(T.load(location))

