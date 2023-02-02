import gym
import random
import numpy as np
import time
from collections import deque
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
        done = False
        obs = env.reset()
        while not done:

            obs_past = obs

            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs_past, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if not done:
                q_past = Q_table[(obs_past, action)]
                td = reward + (DISCOUNT_FACTOR * np.max(np.array([Q_table[(obs, action)] for action in range(env.action_space.n)]))) - Q_table[(obs_past, action)]
                Q_table[(obs_past, action)] = q_past + (LEARNING_RATE * td)

            else:
                q_past = Q_table[(obs_past, action)]
                td = reward - Q_table[(obs_past, action)]
                Q_table[(obs_past, action)] = q_past + (LEARNING_RATE * td)

        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY
            

        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







