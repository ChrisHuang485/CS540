from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
        done_1 = False
        done_2 = False
        obs_2 = env.reset()
        while not done_1 and not done_2:

            obs_past = obs_2

            if random.uniform(0, 1) < EPSILON:
                action_1 = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs_past, i)] for i in range(env.action_space.n)])
                action_1 = np.argmax(prediction)

            obs_1, reward_1, done_1, info_1 = env.step(action_1)
            episode_reward += reward_1

            if random.uniform(0, 1) < EPSILON:
                action_2 = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs_1, i)] for i in range(env.action_space.n)])
                action_2 = np.argmax(prediction)

            obs_2, reward_2, done_2, info_2 = env.step(action_2)
            episode_reward += reward_2

            q_past = Q_table[(obs_past, action_1)]
            td = reward_1 + (DISCOUNT_FACTOR * Q_table[(obs_1, action_2)]) - Q_table[(obs_past, action_1)]
            Q_table[(obs_past, action_1)] = q_past + (LEARNING_RATE * td)

            if done_1 or done_2:
                curr_q = Q_table[(obs_1, action_2)]
                td = reward_2 - Q_table[(obs_1, action_2)]
                Q_table[(obs_1, action_2)] = curr_q + (LEARNING_RATE * td)

        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################


