import numpy as np
import unreal_engine as ue
import pickle


class Game:
    def __init__(self):
        self.actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Attack1', 6: 'Attack2', 7: 'Attack3', 8: 'disengage', 9: 'Idle'}
        self.opponent_actions = {0: 'Move_forward', 1: 'Move_backwards', 2: 'Move_right', 3: 'Move_left', 4: 'Move_uright', 5: 'Move_uleft', 6: 'Move_bright', 7: 'Move_bleft', 8: 'Dodgeforward', 9: 'DodgeRight', 10: 'DodgeLeft', 11: 'DodgeBack', 12: 'dodge_uright', 13: 'dodge_uleft', 14: 'dodge_bright', 15: 'dodge_bleft', 16: 'Attack',17: 'P_Moving_Attack',18: 'R_Moving_Attack', 19: 'Attack1', 20: 'Attack2', 21: 'Attack3', 22: 'disengage', 23: 'Idle'}
        self.npc_hps = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12, 65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.opp_hps = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12, 65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.npc_stmn = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12, 65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.opp_stmn = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12,65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.distances = {0: 'inf - 1000', 1: '1000 - 500', 2: '500 - 300', 3: '300 - 200', 4: '200 - 0'}
        self.NPC_wins = 0
        self.opp_wins = 0
        self.opp_ce_actions = []
        self.NPC_ce_actions = []
        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.episodes = 200
        self.learn_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.9
        self.eps_decay = 0.75
        self.decay_every = (5 / 100) * self.episodes
        self.decay_from = (10 / 100) * self.episodes
        self.iterator = 0
        self.moves_counter = 0
        self.is_attacking = False
        self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.npc_stmn), len(self.opp_stmn), len(self.actions), len(self.opponent_actions), len(self.opponent_actions), len(self.actions)))
        # print(q_table2.ndim)
        # print(q_table2.size)
        # print(q_table2.shape)
        self.q_table2[:, :, :2, :, :, :, :, :, 1:9] = -100  # !Magdi!#
        self.q_table2[:, :, :, 0:10, :, :, :, :, 1:4] = -int('inf')
        self.q_table2[:, :, :, 0:20, :, :, :, :, 4:8] = -int('inf')
        self.q_table2[:, :, :, 0:30, :, :, :, :, 4:8] = -int('inf')
        self.q_table2[:, :, :, 0:45, :, :, :, :, 4:8] = -int('inf')
        self.q_table2[:, :, 0:4, :, :, :, :, :, 0] = 100
        self.q_table2[:, :, 5, :, :, :, :, :, 4:9] = 100
        self.q_table2[:, :, 4, :, :, :, :, :, 4:9] = 5
        self.q_table2[:, :, 3, :, :, :, :, :, 4:9] = 1
        # ue.print_string("Q_Table Class : Constructor")

    def intialize_states(self, cur_old_e_o_hp_dist):
        ue.print_string(f"Iterator :=> {self.iterator}, Epsilon :=> {self.epsilon}")
        if self.iterator == self.episodes:
            return -1

        L = cur_old_e_o_hp_dist.split(',')
        curr_npc_hp = int(L[0])
        curr_opp_hp = int(L[1])
        curr_dist = int(L[2])
        old_npc_hp = int(L[3])
        old_opp_hp = int(L[4])
        old_dist = int(L[5])
        self.opp_ce_actions.append(L[6])
        if L[6] == "true":
            self.is_attacking = True
        else:
            self.is_attacking = False

        # ue.print_string(f"Iterator :=> {self.iterator}, Epsilon :=> {self.epsilon} ,player is attacking is {self.is_attacking}")
        if curr_npc_hp <= 0:
            curr_npc_hp = 0
        if curr_opp_hp <= 0:
            curr_opp_hp = 0
        if old_npc_hp <= 0:
            old_npc_hp = 0
        if old_opp_hp <= 0:
            old_opp_hp = 0

        state = (curr_npc_hp, curr_opp_hp, curr_dist, old_npc_hp, old_opp_hp, old_dist)
        old_distance_index = 0
        if state[5] > 1000:
            old_distance_index = 0
        elif 1000 >= state[5] > 500:
            old_distance_index = 1
        elif 800 >= state[5] > 300:
            old_distance_index = 2
        elif 500 >= state[5] > 200:
            old_distance_index = 3
        elif 200 >= state[5] > 0:
            old_distance_index = 4  # a&a

        new_distance_index = 0
        if state[5] > 1000:
            new_distance_index = 0
        elif 1000 >= state[2] > 500:
            new_distance_index = 1
        elif 800 >= state[2] > 300:
            new_distance_index = 2
        elif 500 >= state[2] > 200:
            new_distance_index = 3
        elif 200 >= state[2] > 0:
            new_distance_index = 4  # a&a

        old_state = (self.npc_hps[state[3]], self.opp_hps[state[4]], old_distance_index, -1, -1, self.NPC_ce_actions[self.NPC_ce_actions.size-1], self.opp_ce_actions[self.opp_ce_actions.length-1], self.opp_ce_actions[self.opp_ce_actions.length-2])
        new_state = (self.npc_hps[state[0]], self.opp_hps[state[1]], new_distance_index, 9, self.opp_ce_actions[self.opp_ce_actions.length-2], self.opp_ce_actions[self.opp_ce_actions.length-3])
        # ue.print_string(f"Old State {old_state}, New State {new_state}")

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state)

        return action

    def take_action(self, index_state):

        if np.random.random() > self.epsilon:
            ue.print_string("Take Max Q_Value")
            action = np.argmax(self.q_table2[index_state])
            self.NPC_ce_actions.append(action)
            ue.log(f"{self.q_table2[index_state]}")
        else:
            ue.print_string("Explore: Random Action")
            action = np.random.randint(0, 9)
            self.NPC_ce_actions.append(action)
        return action

    def next_iterator_epsilon(self, name):
        self.iterator += 1
        self.moves_counter = 0
        self.opp_ce_actions = []
        self.NPC_ce_actions = []
        # ue.print_string(f"MOVES ARE ZEROED YO!!!! {self.moves_counter}")
        if self.iterator % self.decay_every == 0 and self.iterator >= self.decay_from:
            self.epsilon *= self.eps_decay
            ue.print_string("DECAY")

        self.save_table(name)

    def calc_reward(self, current_state, old_state):
        if old_state[1] * 5 - current_state[1] * 5 == 0:
            self.moves_counter += 1
        action = self.NPC_ce_actions(self.NPC_ce_actions.size-1)
        succ_dodge = 0
        # ue.print_string(f"player is attacking is {self.is_attacking}")
        if self.is_attacking is True and old_state[2] == 3 and (action == 1 or action == 2 or action == 3) and \
                old_state[0] * 5 - current_state[0] * 5 == 0:
            succ_dodge = 5
            # ue.print_string(f"successful dodge {self.is_attacking}")
            # ue.log(f"successful dodge , with action {action},#moves {self.moves_counter}")

        reward = (old_state[1] * 5 - current_state[1] * 5) - (old_state[0] * 5 - current_state[0] * 5) - (
                    self.moves_counter * 0.22) + succ_dodge
        # ue.print_string(f"Reward: {reward}, with action {action} ,#moves {self.moves_counter}")

        if old_state[1] * 5 - current_state[1] * 5 != 0:
            self.moves_counter = 0

        max_future_q = np.max(self.q_table2[old_state])
        current_q = self.q_table2[old_state][action]
        new_q = (1 - self.learn_rate) * current_q + self.learn_rate * (reward + self.discount * max_future_q)
        # ue.log(f"Current_q {current_q} in state {old_state} and action {action} => New_q {new_q} in state {current_state}")
        self.q_table2[old_state][action] = new_q

    def take_opponent_action(self, action):
        self.opp_ce_actions.append(action)
        if action == "16" or action == "17" or action == "18" or action == "19" or action == "20" or action == "21":
            self.is_attacking = True
        else:
            self.is_attacking = False

    def save_table(self, name):
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        filename = rf'./Q_Table{args[0]}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table2, f)

            ue.print_string(f"{self.iterator} : Saved Q_Table")
        es = (self.iterator, self.epsilon, self.NPC_wins, self.opp_wins)
        filename = rf'./Episode{args[0]}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(es, f)
            ue.log(f"{self.iterator} : Saved Episode , winning rate : {(self.NPC_wins / self.iterator) * 100}")

    def load_table(self, name):
        filename = rf'./Q_Table{name}.pickle'
        with open(filename, 'rb') as f:
            self.q_table2 = pickle.load(f)

            ue.print_string(f"{self.iterator} : Q_Table is loaded")
        filename = rf'./Episode{name}.pickle'
        with open(filename, 'rb') as f:
            es = pickle.load(f)
            self.iterator = es[0]
            self.epsilon = es[1]
            self.NPC_wins = es[2]
            self.opp_wins = es[3]
            ue.print_string(f"{self.iterator} : #Episodes is loaded")
            ue.log(f"{self.q_table2[:, :, :, 4]}")
            str = f"{self.iterator},{self.NPC_wins},{self.opp_wins}"
        return str
