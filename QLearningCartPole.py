# Q-learning is a specific TD (Temporal-difference) algorithm used to learn the Q-function
# https://en.wikipedia.org/wiki/Inverted_pendulum
# environment: https://github.com/openai/gym/wiki/CartPole-v0
# 2 actions: 0:push_left, 1:push_right
# 4 observations: 0:cart_position ; 1:cart_volecity ; 2:pole_angle; 3:pole_volecity_at_tip

import gym
import numpy as np
import random
import math



# initialize the "Cart-Pole" environment
environment_name = 'CartPole-v0'
environment = gym.make(environment_name)
environment.seed(0)


# 4 observations: 0:cart_position ; 1:cart_volecity ; 2:pole_angle; 3:pole_volecity_at_tip
# number of discrete states  per state dimension
number_states = (1, 1, 6, 3)  # (x, x', theta, theta')

# 2 actions: 0:push_left, 1:push_right
# number of discrete actions
number_actions = environment.action_space.n # (left, right)

# bounds for each discrete state
state_bounds = list(zip(environment.observation_space.low, environment.observation_space.high))
state_bounds[1] = [-1, 1]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# simulation related constants
max_iteration = 1000
max_step = 250
success_to_end = 100
pretest_number = 199

# learning related constants
min_explore_rate = 0.01
min_learning_rate = 0.1

# create qTable with zeros
q_table = np.zeros(number_states + (number_actions,))

def observation_to_state(observation):
    states_list = []
    for i in range(len(observation)):
        if observation[i] <= state_bounds[i][0]:
            state_index = 0
        elif observation[i] >= state_bounds[i][1]:
            state_index = number_states[i] - 1
        else:
            # map the state bounds to the state array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (number_states[i]-1)*state_bounds[i][0]/bound_width
            scaling = (number_states[i]-1)/bound_width
            state_index = int(round(scaling*observation[i] - offset))
        states_list.append(state_index)
    return tuple(states_list)

def select_action(state, explore_rate):
    # select a random action
    if random.random() < explore_rate:
        action = environment.action_space.sample()
    # select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def get_explore_rate(t):
    return max(min_explore_rate, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((t+1)/25)))



if __name__ == "__main__":

    # initial learning parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.98

    num_success = 0

    # training for maximum iteration episodes
    for i in range(max_iteration):

        # reset the environment
        observation = environment.reset()
        total_reward = 0
        # the initial state
        state_0 = observation_to_state(observation)

        # each episode is max_step long
        for t in range(max_step):
            environment.render()
            # select an action
            action = select_action(state_0, explore_rate)
            # get observation, reward and done after each step, execute the action
            observation, reward, done, _ = environment.step(action)
            # observe the result
            state = observation_to_state(observation)
            # update the Q table
            # Bellmann eq: Q(s,a)=reward + discount_factor* max(Q(s_,a_))  ::: Q_target = reward+discount_factor*max(Qs_prime)
            # TD_target=(reward + discount_factor * (best_q)) 
            # TD_error=(reward + discount_factor * (best_q) - q_table[state_0 + (action,)]) 
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] = q_table[state_0 + (action,)] + learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            # setting up for the next iteration
            state_0 = state

            total_reward += reward

            if done:
                print('Iteration No: %d -- TimeSteps:%d -- Success: %d -- Best Q: %f --Explore rate: %f --Learning rate: %f --Total reward: %d' % (i + 1, t, num_success,best_q,explore_rate,learning_rate,total_reward))
                # after pretest_number
                if (t >=pretest_number):
                    num_success += 1
                else:
                    num_success = 0
                break

        # break when it's solved over num_success=100 times consecutively
        if num_success > success_to_end:
            break
        # update parameters
        explore_rate = get_explore_rate(i)
        learning_rate = get_learning_rate(i)
