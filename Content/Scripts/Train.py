import numpy as np
import unreal_engine as ue
import pickle


class Game:
    def __init__(self):
        self.actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Idle'}
        self.play_healths = {100: 0, 95: 1, 90: 2, 85: 3, 80: 4, 75: 5, 70: 6, 65: 7, 60: 8, 55: 9, 50: 10, 45: 11, 40: 12, 35: 13, 30: 14, 25: 15, 20: 16, 15: 17, 10: 18, 5: 19, 0: 20}
        self.enemy_healths = {100: 0, 95: 1, 90: 2, 85: 3, 80: 4, 75: 5, 70: 6, 65: 7, 60: 8, 55: 9, 50: 10, 45: 11, 40: 12, 35: 13, 30: 14, 25: 15, 20: 16, 15: 17, 10: 18, 5: 19, 0: 20}
        self.distances = {0: '1500 - 800', 1: '800 - 500', 2: '500 - 200', 3: '200 - 0'}

        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.episodes = 200
        self.learn_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.9
        self.eps_decay = 0.95
        self.decay_every = (5/100) * self.episodes
        self.decay_from = (10/100) * self.episodes
        self.iterator = 0
        self.moves_counter = 0
        self.is_attacking = False

        self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.play_healths), len(self.enemy_healths), len(self.distances), len(self.actions)))
        # print(q_table2.ndim)
        # print(q_table2.size)
        # print(q_table2.shape)
        self.q_table2[:, :, :2, 4] = -float('inf') #!Magdi!#
        self.q_table2[:, :, 3, 4] = 10
        self.q_table2[:, :, 2, 4] = 3
        ue.print_string("Q_Table Class : Constructor")

    def intialize_states(self, cur_old_e_o_hp_dist):
        ue.print_string(f"Iterator :=> {self.iterator}, Epislon :=> {self.epsilon}")
        if self.iterator == self.episodes:
            return -1
    
        L = cur_old_e_o_hp_dist.split(',')
        c_p_health = int(L[0])
        c_e_health = int(L[1])
        c_dist = int(L[2])
        o_p_health = int(L[3])
        o_e_health = int(L[4])
        o_dist = int(L[5])
        self.is_attacking = bool(L[6])

        if c_p_health <= 0:
            c_p_health = 0
        if c_e_health <= 0:
            c_e_health = 0
        if o_p_health <= 0:
            o_p_health = 0
        if o_e_health <= 0:
            o_e_health = 0

        state = (c_p_health, c_e_health, c_dist, o_p_health, o_e_health, o_dist)
        old_distance_index = 0
        if 1500 >= state[5] > 800:
            old_distance_index = 0
        elif 800 >= state[5] > 500:
            old_distance_index = 1
        elif 500 >= state[5] > 200:
            old_distance_index = 2
        elif 200 >= state[5] > 0:
            old_distance_index = 3

        new_distance_index = 0
        if 1500 >= state[2] > 800:
            new_distance_index = 0
        elif 800 >= state[2] > 500:
            new_distance_index = 1
        elif 500 >= state[2] > 200:
            new_distance_index = 2
        elif 200 >= state[2] > 0:
            old_distance_index = 3

        old_state = (self.play_healths[state[0]], self.enemy_healths[state[1]], old_distance_index)
        new_state = (self.play_healths[state[3]], self.enemy_healths[state[4]], new_distance_index)
        ue.print_string(f"Old State {old_state}, New State {new_state}")

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state[0], new_state[1], new_state[2])
        
        return action

    def take_action(self, p_health, e_health, d_stance):
        index_state = (p_health, e_health, d_stance)

        if np.random.random() > self.epsilon:
            ue.print_string("Take Max Q_Value")
            action = np.argmax(self.q_table2[index_state])
        else:
            ue.print_string("Explore: Random Action")
            action = np.random.randint(0, 6)
        return action

    def next_iterator_epsilon(self):
        self.iterator += 1
        if self.iterator % self.decay_every == 0 and self.iterator > self.decay_from:
            self.epsilon *= self.eps_decay
            
        self.save_table()
        
    def calc_reward(self, current_state, old_state):
        if old_state[0] - current_state[0] == 0:
            self.moves_counter += 1

        succ_dodge = 0
        if self.is_attacking and current_state[2] >= 200:
            succ_dodge = 15

        reward = (old_state[0] - current_state[0]) - (old_state[1] - current_state[1]) - (self.moves_counter * 0.22) + succ_dodge

        if old_state[0] - current_state[0] != 0:
            self.moves_counter = 0

        action = self.take_action(old_state[0], old_state[1], old_state[2])
        max_future_q = np.max(self.q_table2[old_state])
        current_q = self.q_table2[old_state][action]
        new_q = (1 - self.learn_rate) * current_q + self.learn_rate * (reward + self.discount * max_future_q)
        ue.log(f"Current_q {current_q} in state {old_state} and action {action} => New_q {new_q} in state {current_state}")
        self.q_table2[old_state][action] = new_q

    def save_table(self):
        filename = r'Q_Table.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table2, f)
            ue.print_string(f"{self.iterator} : Saved Q_Table")
        filename = r'Episode.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self.iterator, f)
            ue.print_string(f"{self.iterator} : Saved Episode")

    def load_table(self):
        filename = r'Q_Table.pickle'
        with open(filename, 'rb') as f:
            self.q_table2 = pickle.load(f)
            ue.print_string(f"{self.iterator} : Saved Q_Table")
        filename = r'Episode.pickle'
        with open(filename, 'rb') as f:
            self.iterator = pickle.load(f)
            ue.print_string(f"{self.iterator} : Load Episode")
        return self.iterator
