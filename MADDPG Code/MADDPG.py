import torch as T
import torch.nn.functional as F
from Agent import Agent
import numpy as np


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 save_folder='test', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.exploration_rate = 1.0
        self.exploration_decay = 0.0002
        self.min_exploration_rate = 0.001
        chkpt_dir += save_folder
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self, folder, iteration, update=""):
        print(update)
        for agent in self.agents:
            agent.save_models(folder, iteration)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):

        actions = []
        # if np.random.uniform() < self.exploration_rate:
        #     for _, _ in enumerate(self.agents):
        #         # print("ra", end="")
        #         action = np.random.uniform(np.array([-1, -1, -1]), np.array([1, 1, 1]), size=3)
        #         actions.append(action)
        # else:
        for agent_idx, agent in enumerate(self.agents):
            # print("ca")
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)

        # self.exploration_rate -= self.exploration_decay
        # self.exploration_rate = self.exploration_rate if self.exploration_rate > self.min_exploration_rate else self.min_exploration_rate

        return actions

    def get_alpha(self, i):
        return self.agents[i].get_alpha()

    def to_tensor(self, pi):
        print(type(pi), len(pi))
        print(pi[0], "\n", type(pi[0]), end="\n\n")
        print(pi[1], "\n", type(pi[1]), end="\n\n")
        print(pi[2], "\n", type(pi[2]), end="\n\n")
        print(pi, end="\n\nnew pi\n\n\n\n")


        action0 = pi[0].detach().cpu().numpy()
        action0 = np.argmax(action0) - 1

        action1 = pi[1].detach().cpu().numpy()
        action1 = np.argmax(action1) - 1

        print(T.tensor([action0, action1, 0]))

        return T.tensor([action0, action1, 0])

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            #we have a tuple of tensors we want a single tensor similar to what we would get from softmax
            #here is what we will do we shall

            all_agents_new_actions.append(self.to_tensor(new_pi))
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(self.to_tensor(pi))
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()  # r not converted stops here...
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
