import numpy as np
import unreal_engine as ue


class Game:
    def __init__(self):
        self.actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Idle'}
        self.play_healths = {0: 0, 10: 1, 25: 2, 40: 3, 55: 4, 70: 5, 85: 6, 100: 7}
        self.enemy_healths = {0: 0, 10: 1, 25: 2, 40: 3, 55: 4, 70: 5, 85: 6, 100: 7}
        self.distances = {0: '1100 - 800', 1: '800 - 500', 2: '500 - 200', 3: '200 - 0'}

        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.learn_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.9
        # eps_decay = 0.5

        self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.play_healths), len(self.enemy_healths), len(self.distances), len(self.actions)))
        # print(q_table2.ndim)
        # print(q_table2.size)
        # print(q_table2.shape)
        self.q_table2[:, :, 0:2, 4:6] = -float('inf') #!Magdi!#


    def intialize_states(self, cur_old_e_o_hp_dist):
        L = cur_old_e_o_hp_dist.split(',')
        c_p_health = int(L[0])
        c_e_health = int(L[1])
        c_dist     = int(L[2])
        o_p_health = int(L[3])
        o_e_health = int(L[4])
        o_dist     = int(L[5])

        state = (c_p_health, c_e_health, c_dist, o_p_health, o_e_health, o_dist)
        old_distance_index = 0
        if 1100 >= state[5] > 800:
            old_distance_index = 0
        elif 800 >= state[5] > 500:
            old_distance_index = 1
        elif 500 >= state[5] > 200:
            old_distance_index = 2
        elif 200 >= state[5] > 0:
            old_distance_index = 3

        new_distance_index = 0
        if 1100 >= state[2] > 800:
            new_distance_index = 0
        elif 800 >= state[2] > 500:
            new_distance_index = 1
        elif 500 >= state[2] > 200:
            new_distance_index = 2
        elif 200 >= state[2] > 0:
            old_distance_index = 3

        old_state = (self.play_healths[state[0]], self.enemy_healths[state[1]], old_distance_index)
        new_state = (self.play_healths[state[3]], self.enemy_healths[state[4]], new_distance_index)

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state[0], new_state[1], new_state[2])
        return action

    def take_action(self, p_health, e_health, d_stance):
        index_state = (p_health, e_health, d_stance)
        #action = np.argmax(self.q_table2[index_state])
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table2[index_state])
            print(action)
        else:
            action = np.random.randint(0, 6)
        return action

    def calc_reward(self, current_state, old_state):
        reward = (old_state[1] - current_state[1]) - (old_state[0] - current_state[0])

        action = self.take_action(old_state[0], old_state[1], old_state[2])
        max_future_q = np.max(self.q_table2[old_state])
        current_q = self.q_table2[old_state][action]
        new_q = (1 - self.learn_rate) * current_q + self.learn_rate * (reward + self.discount * max_future_q)
        ue.log(f"Current_q {current_q} in state {old_state} and action {action} => New_q {new_q} in state {current_state}")
        self.q_table2[old_state][action] = new_q
