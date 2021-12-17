import math

from dm_control.locomotion import soccer
from dm_control import viewer
import numpy as np
import torch as T
from Network import Network


def gen_policy(f1, f2, f3, f4, nc1=True, nc2=True, nc3=True, nc4=True):
    def manual_policy(time_step):
        obs = convertObservation(time_step)
        actions = []
        actions.append(f1(obs[0]) if nc1 else f1(time_step.observation[0]))
        actions.append(f2(obs[1]) if nc2 else f2(time_step.observation[1]))
        actions.append(f3(obs[2]) if nc3 else f3(time_step.observation[2]))
        actions.append(f4(obs[3]) if nc4 else f4(time_step.observation[3]))
        return actions

    return manual_policy


def calculate_distance_reward(x1, y1, x2, y2, value):
    distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return value / distance


def get_no_op():
    def no_op(timestep):
        del timestep
        return np.array([0, 0, 0])

    return no_op


def get_manual(action):
    def x(timestep):
        del timestep
        return np.array(action)

    return x


def get_nn_policy(folder, i, subfolder, epoch):
    save_dir = "results" + folder
    actor = Network(None, i, save_dir=save_dir)
    actor.load_checkpoint(subfolder, epoch)

    def policy(obs):
        state = T.tensor([obs], dtype=T.float)
        v, m, t, j = actor.forward(state)
        return get_action(m, t, j)

    return policy


def get_action(m, t, j):
    options = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    m_action, t_action, j_action = get_actions(m, t, j)
    return np.array([
        options[int(m_action.numpy())],
        options[int(t_action.numpy())],
        options[int(j_action.numpy())]
    ])


def get_actions(m, t, j):
    m_prob = T.distributions.Categorical(m)
    m_action = m_prob.sample()
    t_prob = T.distributions.Categorical(t)
    t_action = t_prob.sample()
    j_prob = T.distributions.Categorical(j)
    j_action = j_prob.sample()
    return m_action, t_action, j_action


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


def get_run_to_ball():
    def get_action_to_ball(agent_obs):
        # agent_obs = obs.observation[i]
        if need_to_turn(agent_obs):
            # print("turning")
            action = turn(agent_obs)
        else:
            # print("running")
            action = [1, 0, 0]
        return np.array(action)

    return get_action_to_ball


def need_to_turn(obs):
    _, ball_y, _ = obs["ball_ego_position"][0]
    if abs(ball_y) < 1.0:
        return False
    else:
        return True


def turn(obs):
    _, ball_y, _ = obs["ball_ego_position"][0]
    return [0, 1, 0] if ball_y > 0 else [0, -1, 0]


def valid_spec(spec):
    if spec.startswith("stats"): return False
    if spec.startswith("teammate"): return False
    if spec.startswith("opponent"): return False
    if spec.startswith("prev"): return False
    if spec.startswith("body"): return False
    if spec.startswith("joints"): return False
    if spec.startswith("sensors"): return False
    if spec.startswith("world"): return False
    return True


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


def get_num_actions(env):
    return env.action_spec()[0].shape[0]


def main():
    env = soccer.load(team_size=2,
                      time_limit=10000.0,
                      disable_walker_contacts=False,
                      enable_field_box=True,
                      terminate_on_goal=False,
                      walker_type=soccer.WalkerType.BOXHEAD)

    # Change this between the different folders to select which weights to load this works for finalRun, fullTraining, and completeRewrite
    FOLDER = "/" + "/finalRun"

    # Change this between "/bestAverge", "bestScore", and "/iter" to select which weights to load
    subfolder = "/bestAverage"

    # Change the epoch to match the iteration in the desired weights file
    #these settings load the best average weights for the finalTraining run
    f1, nc1 = get_nn_policy(FOLDER, 0, subfolder, epoch=64583), True
    f3, nc3 = get_nn_policy(FOLDER, 1, subfolder, epoch=6717), True

    # f1, nc1 = get_run_to_ball(), False
    # f2, nc2 = get_run_to_ball(), False
    # f3, nc3 = get_run_to_ball(), False
    # f4, nc4 = get_run_to_ball(), False

    # f1, nc1 = get_manual([0, 0, 0]), False
    # f2, nc2 = get_manual([0, 0, 0]), False
    # f3, nc3 = get_manual([0, 0, 0]), False
    # f4, nc4 = get_manual([0, 0, 0]), False

    # f1, nc1 = get_no_op(), False
    f2, nc2 = get_no_op(), False
    # f3, nc3 = get_no_op(), False
    f4, nc4 = get_no_op(), False

    # Launch the viewer application.
    viewer.launch(env, policy=gen_policy(f1, f2, f3, f4, nc1=nc1, nc2=nc2, nc3=nc3, nc4=nc4))


main()
