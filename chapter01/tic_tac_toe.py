# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# %%
import numpy as np
import pickle 
import pprint as pp

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

# %%
class State:
    def __init__(self):
        # the board is represented by an n * n array, 
        # 1 represents a chessman of the player who moves first, 
        # -1 represents a chessman of another player
        # 0 represents and empy position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None 
        self.end = None

    # compute the hash value for one state, it's unique
    # ref. hash in python https://stackoverflow.com/questions/17585730/what-does-hash-do-in-python
    # ref hash function https://en.wikipedia.org/wiki/Hash_function
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :])) # add numbers in each row
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        #check diagonals 
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i] #the other diagonal
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        
        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self,  i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board 
    def print_state(self):
        for i in range(BOARD_ROWS):
            print("------------")
            out = "| "
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = "*"
                elif self.data[i, j] == -1:
                    token = "x"
                else:
                    token = "0"
                out += token + " | "
            print(out)
        print("------------")

# get all configurations of the board recursively
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)

def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# all possible board configurations
all_states = get_all_states()

# %%
s = State()
s.print_state()

# %%
s_new = s.next_state(0, 0, 1)
s_new.print_state()

# %%
s_new.is_end()
# %%
all_states

# %%
all_states[16402]

# %%
all_states[16402][0].print_state()

# %%
all_states[14764]

# %%
all_states[14764][0].print_state()

# %%
len(all_states)

# %%
# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    #set up initial values
    #@init_est: the initial probablity of winning for tie
    """
    Assuming we always play Xs, then for all states with three Xs
    in a row the probability of winning is 1, because we have already 
    won. Similarly, for all states with three Os in a row, or that are filled 
    up, the correct probability is 0, as we cannot win from them. 
    We set the initial values of all the other states to 0.5, representing 
    a guess that we have a 50% chance of winning.(intro to RL by Sutton p9)
    """
    def set_symbol(self, symbol, init_est=0.5):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = init_est
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5
        
    # update value estimation
    # @exp_included: include exploratory moves in learning
    def backup(self, exp_included=False):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]

            if not exp_included:
                td_error = self.greedy[i] * ( # greedy consists of True or False(exploratory)
                    self.estimations[states[i + 1]] - self.estimations[state]
                )
            else:
                td_error = self.estimations[states[i + 1]] - self.estimations[state]
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        # expolatory move
        """
        we select randomly from among the other moves instead. 
        These are called exploratory moves because they cause 
        us to experience states that we might otherwise never see.
        (intro to RL by Sutton p9)
        """
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        # ref. "Sorts are guaranteed to be stable. That means that when 
        # multiple records have the same key, their original order is preserved.""
        # https://docs.python.org/3/howto/sorting.html#sortinghowto
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True) # the largest, the first
        action = values[0][1]
        action.append(self.symbol)
        return action

    def give_policy(self):
        return self.estimations

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# %%
p1 = Player(step_size=0.1, epsilon=0.1)
p1.estimations

# %%
p1.set_symbol(1)

# %%
p1.estimations

# %%
all_states[15890][0].print_state()


# %%
class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    # @init_est: inital estimation of probability of winning for tie
    def __init__(self, player1, player2, init_est=0.5):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None 
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol, init_est)
        self.p2.set_symbol(self.p2_symbol, init_est)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2
    
    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner

# %%

# %%
p1 = Player()
p2 = Player()
judger = Judger(p1, p2)
judger.play(print_state=True)
# %%


# %%
# human interface 
# input a number to pua chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol, init_est):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key) #keys = ['q', 'w', 'e', ...]
        i = data // BOARD_COLS 
        j = data % BOARD_COLS
        return i, j, self.symbol

# %%
# @init_est: the initial probablity of winning for tie
# @exp_included: include exploratory moves in learning
def train(epochs, print_every_n=500, print_policy=False, init_est=0.5, exp_included=False):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2, init_est)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()
    if print_policy:
        pp.pprint(player1.estimations)
    return (player1.give_policy(), player2.give_policy())

# %%
train(5000)

# %%
def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

# %%
compete(5000)

# %%
# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie. 
# So we test weather the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")

# %%
#train(int(1e4))
#compete(int(1e3))
play()

# %% [markdown]

### Let's change inital estimation of probability of winning for tie
# %% 
# original
player1_est_orig = train(int(1e5), init_est=0.5)[0]
player1_est_zero = train(int(1e5), init_est=0)[0]
# %%
est_diff = {}
for i in player1_est_orig.keys():
    est_diff[i] = player1_est_orig[i] - player1_est_zero[i]

pp.pprint(est_diff)
# %% [markdown]

## Let's include exploratory moves into learning

# %%
player1_est_exp = train(int(1e5), init_est=0.5, exp_included=True)[0]

# %%
est_diff_exp = {}
for i in player1_est_orig.keys():
    est_diff[i] = player1_est_orig[i] - player1_est_exp[i]

pp.pprint(est_diff)

