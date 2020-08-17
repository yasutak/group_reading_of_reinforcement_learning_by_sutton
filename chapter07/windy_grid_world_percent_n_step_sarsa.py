#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# %%
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


# %%
# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# %%
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

# %%
# play for an episode
def one_step_sarsa_episode(q_value):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time

def n_step_sarsa_episode(q_value, n):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    states = [state]
    actions = []

    T = float('inf')
    time = 0

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    actions.append(action)

    while True:
        time += 1

        if time < T:
            
            next_state = step(state, action)
            states.append(next_state)
            
            if next_state == [3, 7]:
                T = time + 1
            else:
                if np.random.binomial(1, EPSILON) == 1:
                    next_action = np.random.choice(ACTIONS)
                else:
                    values_ = q_value[next_state[0], next_state[1], :]
                    next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
                    action = next_action
                actions.append(next_action)

        update_time = time - n

        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += -1
            # add state value to the return
            if update_time + n + 1 < T:
                #print(states[update_time + n][0], states[update_time + n][1], actions[update_time + n])
                returns += float(q_value[states[update_time + n][0], states[update_time + n][1], actions[update_time + n]])
            state_to_update = states[update_time]
            # update the state value
            if not state_to_update == [3, 7]:
                q_value[state_to_update[0], state_to_update[1], actions[update_time]] += ALPHA * (returns - q_value[state_to_update[0], state_to_update[1], actions[update_time]])
            if update_time == T - 1:
                break
        state = next_state
        
    return time


# %%
def figure_6_3():

    ns = np.arange(1, 10, 1)
    for n in ns:
        q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        episode_limit = 500

        steps = []
        ep = 0
        while ep < episode_limit:
            steps.append(n_step_sarsa_episode(q_value, n))
            # time = episode(q_value)
            # episodes.extend([ep] * time)
            ep += 1

        steps = np.add.accumulate(steps)

        plt.plot(steps, np.arange(1, len(steps) + 1), label='n = %d' % n)

    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.show()
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


# %%
figure_6_3()




# %%
