import math
import os
import sys

import numpy as np
import torch.optim
from dm_control.locomotion import soccer
import matplotlib.pyplot as plt
import torch as T

from Network import Network

"""
@:param observation a list of np.arrays representing each agent's observation space
@returns the observations concatenated together as an np.array
"""
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


"""
@param i the index of the agent in question
@param env the environment
@returns the size of the observation space as an integer
"""
def get_observation_size(i, env):
    x = 0
    observation_spec = env.observation_spec()[i]
    for spec in observation_spec:
        if valid_spec(spec):
            if len(observation_spec[spec].shape) > 1:
                x += observation_spec[spec].shape[1]
            else:
                x += observation_spec[spec].shape[0]
    return x


"""
@param env the environment
@returns the size of the actionspace
"""
def get_num_actions(env):
    return env.action_spec()[0].shape[0]


"""
@param timestep and object with data representing the current state of an environment
@returns a list of np.arrays representing the obervation spaces for each of the agents
"""
def convertObservation(timestep):
    obs = []
    for observation in timestep.observation:
        actor_obs = []
        for spec in observation:
            if valid_spec(spec):
                for i in observation[spec]:
                    if type(i) is np.float64:
                        actor_obs.append(i)
                    else:
                        for j in i:
                            actor_obs.append(j)
        obs.append(np.array(actor_obs))
    return obs


"""
@param spec the string representing the name of a potential parameter in the environment
@returns a boolean representing whether the parameter should be considered for training and action choice
"""
def valid_spec(spec):
    # if spec.startswith("ball"): return True
    if spec.startswith("stats"): return False
    if spec.startswith("teammate"): return False
    if spec.startswith("opponent"): return False
    if spec.startswith("prev"): return False
    if spec.startswith("body"): return False
    if spec.startswith("joints"): return False
    if spec.startswith("sensors"): return False
    if spec.startswith("world"): return False
    # print(spec)
    return True

"""
@param folder the name of the folder

creates empty folders for the sake of storing graphs and weights
"""
def makeFolders(folder):
    # make the results/folder/graphs
    makefolder("results" + folder + "/graphs")
    # make the results/folder/weights/bestAverage
    makefolder("results" + folder + "/weights/bestAverage")
    # make the results/folder/weights/bestScore
    makefolder("results" + folder + "/weights/bestScore")
    # make the results/folder/weights/iter
    makefolder("results" + folder + "/weights/iter")
    pass


"""
@:param path a string representing a file path
creates the path if it does not yet exist
"""
def makefolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


"""
:param obs an observation representing information about the current timestep of the environment
:returns
"""
def convertActions(obs):
    x = []
    x.append(get_action_to_ball(obs, 0))
    x.append(np.array([0, 0, 0]))

    return x

"""
a manual function to get the best action to cause an agent to run straight toward a ball
"""
def get_action_to_ball(obs, i):
    agent_obs = obs.observation[i]
    if need_to_turn(agent_obs):
        action = turn(agent_obs)
    else:
        action = [1, 0, 0]
    return np.array(action)

"""
determines of the agent needs to turn in order to run toward the ball
"""
def need_to_turn(obs):
    _, ball_y, _ = obs["ball_ego_position"][0]
    if abs(ball_y) < 1:
        return False
    else:
        return True

"""
determines the action needed to turn
"""
def turn(obs):
    _, ball_y, _ = obs["ball_ego_position"][0]

    return [0, 1, 0] if ball_y > 0 else [0, -1, 0]


"""
converts a timestep object to a next state obs (which is a list of np arrays) reward (list of floats) and done (list of booleans) and a dictionary
"""
def convertTimestep(timestep, prev_timestep):
    obs_ = convertObservation(timestep)
    reward = []
    done = []

    if timestep.reward is not None:
        for i in range(len(timestep.reward)):
            base_reward = timestep.reward[i] * 100
            if base_reward > 0: print(f"huh... {i} scored....")
            ball_x, ball_y, _ = timestep.observation[i]["ball_ego_position"][0]
            distance_delta = 0.0
            if prev_timestep is not None:
                prev_ball_x, prev_ball_y, _ = prev_timestep.observation[i]["ball_ego_position"][0]
                curr_distance = distance(ball_x, ball_y, 0, 0)
                prev_distance = distance(prev_ball_x, prev_ball_y, 0, 0)
                distance_delta = prev_distance - curr_distance
            goal_x, goal_y = timestep.observation[i]["opponent_goal_back_left"][0]
            distance_reward = calculate_distance_reward(ball_x, ball_y, goal_x, goal_y, 0.001)
            ball_reward = calculate_distance_reward(ball_x, ball_y, 0, 0, 1.0)
            reward.append(float(base_reward + distance_delta + distance_reward))
        done = [timestep.last() for _ in range(len(timestep.reward))]
    info = {}

    return obs_, reward, done, info

"""
calculates the reward for being between point x and point 2
"""
def calculate_distance_reward(x1, y1, x2, y2, value):
    distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return value / distance

"""
calculates the distance between two points
"""
def distance(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


"""
used to calculate part of the loss function as well as to get the action for the environment
"""
def log_prob(m1):
    m1_prob = T.distributions.Categorical(m1)
    m1_action = m1_prob.sample()
    m1_log = -m1_prob.log_prob(m1_action)

    return m1_action, m1_log


if __name__ == '__main__':
    FOLDER = "/" + "finalRunA"
    env = soccer.load(team_size=1,
                      time_limit=10.0,
                      disable_walker_contacts=False,
                      enable_field_box=True,
                      terminate_on_goal=False,
                      walker_type=soccer.WalkerType.BOXHEAD)
    save_dir = "results" + FOLDER
    n_agents = len(env.action_spec())
    agents = []
    for i in range(n_agents):
        agents.append(Network(get_observation_size(i, env), i, save_dir=save_dir))
    agent1optimizer = torch.optim.Adam(agents[0].parameters())
    agent2optimizer = torch.optim.Adam(agents[1].parameters())

    PRINT_INTERVAL = 100
    GRAPH_INTERVAL = 1000
    N_GAMES = 500000

    a1h = []
    a2h = []
    best_average_score = 0
    best_score = 0
    a1bs = -1000
    a2bs = -1000
    a1bas = -1000
    a2bas = -1000
    x = []
    x_score = []
    y1 = []
    y2 = []
    agent_1_score = []
    agent_2_score = []

    makeFolders(FOLDER)

    for epoch in range(N_GAMES):
        timestep = env.reset()

        agent_2_score = 0
        agent_1_score = 0
        done = False
        episode_step = 0
        critic1_values = []
        critic2_values = []
        actor1_movements = []
        actor2_movements = []
        actor1_turnings = []
        actor2_turnings = []
        actor1_jumpings = []
        actor2_jumpings = []
        agent1_rewards = []
        agent2_rewards = []
        options = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]

        while not done:
            # get the state      obs_, reward, done, info
            obs, _, _, _ = convertTimestep(timestep, None)

            # get the actions (probabalisitc)
            v1, m1, t1, j1 = agents[0].forward(T.tensor(obs[0], dtype=T.float))
            v2, m2, t2, j2 = agents[1].forward(T.tensor(obs[1], dtype=T.float))

            m1_action, m1_log = log_prob(m1)
            m2_action, m2_log = log_prob(m2)
            t1_action, t1_log = log_prob(t1)
            t2_action, t2_log = log_prob(t2)
            j1_action, j1_log = log_prob(j1)
            j2_action, j2_log = log_prob(j2)

            # store actions and values
            critic1_values.append(v1)
            critic2_values.append(v2)
            actor1_movements.append(m1_log)
            actor2_movements.append(m2_log)
            actor1_turnings.append(t1_log)
            actor2_turnings.append(t2_log)
            actor1_jumpings.append(j1_log)
            actor2_jumpings.append(j2_log)

            # format actions
            action1 = np.array([
                options[int(m1_action.numpy())],
                options[int(t1_action.numpy())],
                options[int(j1_action.numpy())]
            ])
            action2 = np.array([
                options[int(m2_action.numpy())],
                options[int(t2_action.numpy())],
                options[int(j2_action.numpy())]
            ])
            actions = [action1, action2]

            # step the environment
            prev_timestep = timestep
            timestep = env.step(actions)
            _, reward, done, _ = convertTimestep(timestep, prev_timestep)
            done = done[0]

            # store reward
            agent1_rewards.append(reward[0])
            agent2_rewards.append(reward[1])

            # graphing purposes
            agent_1_score += reward[0]
            agent_2_score += reward[1]

            # if done break
            if done: break

        # calculate the discounts i.e. g
        returns1 = []
        returns2 = []
        DISCOUNT = 0.95
        discounted_sum2 = 0
        discounted_sum1 = 0
        for i in range(len(agent1_rewards) - 1, -1, -1):
            discounted_sum1 = agent1_rewards[i] + DISCOUNT * discounted_sum1
            discounted_sum2 = agent2_rewards[i] + DISCOUNT * discounted_sum2
            returns1.insert(0, discounted_sum1)
            returns2.insert(0, discounted_sum2)

        returns1 = T.tensor(returns1)
        returns2 = T.tensor(returns2)

        # initialize the 8 losses
        critic1_loss = []
        actor1_loss = []

        for log1, log2, log3, value, g in zip(actor1_movements, actor1_turnings, actor1_jumpings, critic1_values,
                                              returns1):
            advantage = (g - value)[0]
            actor1_loss.append(((log1 + log2 + log3) / 3) * advantage)
            critic1_loss.append(advantage * advantage)

        critic1_loss = sum(critic1_loss)
        actor1_loss = sum(actor1_loss)
        agent1_loss = critic1_loss + actor1_loss

        critic2_loss = []
        actor2_loss = []

        for log1, log2, log3, value, g in zip(actor2_movements, actor2_turnings, actor2_jumpings, critic2_values,
                                              returns2):
            advantage = (g - value)[0]
            actor2_loss.append(((log1 + log2 + log3) / 3) * advantage)
            critic2_loss.append(advantage * advantage)

        critic2_loss = sum(critic2_loss)
        actor2_loss = sum(actor2_loss)
        agent2_loss = critic2_loss + actor2_loss

        # Apply gradients
        agent1_loss.backward()
        agent1optimizer.step()
        agent2_loss.backward()
        agent2optimizer.step()

        a1h.append(agent_1_score)
        a1as = np.mean(a1h[-100:])
        a2h.append(agent_2_score)
        a2as = np.mean(a2h[-100:])

        if a1as > a1bas and epoch > 100:  # agent 1's best average improved
            print(f"\tagent 1's best average score increased from {a1bas:.4f} to {a1as:.4f}")
            a1bas = a1as
            agents[0].save_checkpoint(subfolder="/bestAverage", epoch=epoch)
        if a2as > a2bas and epoch > 100:  # agent 2's best average improved
            print(f"\tagent 2's best average score increased from {a2bas:.4f} to {a2as:.4f}")
            a2bas = a2as
            agents[1].save_checkpoint(subfolder="/bestAverage", epoch=epoch)
        if agent_1_score > a1bs:  # agent 1's best average improved
            print(f"\tagent 1's best score increased from {a1bs:.4f} to {agent_1_score:.4f}")
            a1bs = agent_1_score
            agents[0].save_checkpoint(subfolder="/bestScore", epoch=epoch)
        if agent_2_score > a2bs:  # agent 2's best average improved
            print(f"\tagent 2's best score increased from {a2bs:.4f} to {agent_2_score:.4f}")
            a2bs = agent_2_score
            agents[1].save_checkpoint(subfolder="/bestScore", epoch=epoch)

        if epoch % PRINT_INTERVAL == 0 and epoch > 0:
            print(f"Episode: {epoch}, agent 1 average score: {a1as:.4f}, agent 2 average score: {a2as}")
            x.append(epoch)
            y1.append(a1as)
            y2.append(a2as)
            if epoch % GRAPH_INTERVAL == 0 and epoch > 0:
                plt.plot(x, y1, label="blue agent" if epoch == GRAPH_INTERVAL else "", color='b')
                plt.plot(x, y2, label="red agent" if epoch == GRAPH_INTERVAL else "", color='r')
                plt.legend()
                plt.xlabel("Iteration")
                plt.ylabel("Average Reward (Previous 100 Epochs)")
                title = "AverageReward_Epoch_" + str(epoch)
                plt.title(title)
                plt.savefig(fname="results" + FOLDER + "/graphs/" + title + ".png")
                agents[0].save_checkpoint("/iter", epoch)
                agents[1].save_checkpoint("/iter", epoch)
