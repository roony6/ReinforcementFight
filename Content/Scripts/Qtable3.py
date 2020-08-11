import numpy as np
import unreal_engine as ue
import pickle


class Game:
    def __init__(self):
        self.actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Attack1', 6: 'Attack2', 7: 'Attack3', 8: 'disengage', 9: 'Idle'}
        self.opponent_actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Attack1', 6: 'Attack2', 7: 'Attack3', 8: 'disengage', 9: 'Idle'}
        #self.opponent_actions = {0: 'Move_forward', 1: 'Move_backwards', 2: 'Move_right', 3: 'Move_left', 4: 'Move_uright', 5: 'Move_uleft', 6: 'Move_bright', 7: 'Move_bleft', 8: 'Dodgeforward', 9: 'DodgeRight', 10: 'DodgeLeft', 11: 'DodgeBack', 12: 'dodge_uright', 13: 'dodge_uleft', 14: 'dodge_bright', 15: 'dodge_bleft', 16: 'Attack',17: 'P_Moving_Attack',18: 'R_Moving_Attack', 19: 'Attack1', 20: 'Attack2', 21: 'Attack3', 22: 'disengage', 23: 'Idle'}
        self.npc_hps = {25: 0, 50: 1, 75: 2, 100: 3}
        self.opp_hps = {25: 0, 50: 1, 75: 2, 100: 3}
        self.npc_stmn = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10}
        self.opp_stmn = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10}
        self.distances = {0: 'inf - 700', 1: '700 - 550', 2: '550 - 300', 3: '300 - 0'}
        self.NPC_wins = 0
        self.opp_wins = 0
        # evaluation criteria
        self.winning_rate = []
        self.d_taken = []
        self.dmg_taken = 0
        self.d_dealt = []
        self.dmg_dealt = 0
        self.move_to_q = []
        self.attack_q = []
        self.dodge_q = []
        ########################
        self.opp_ce_actions = [9, 9, 9, 9]
        self.NPC_ce_actions = [9, 9, 9, 9]
        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.episodes = 2000
        self.learn_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.9
        self.eps_decay = 0.75
        self.decay_every = (5 / 100) * self.episodes
        self.decay_from = (10 / 100) * self.episodes
        self.iterator = 0
        self.moves_counter = 0
        self.is_attacking = False
        self.name = ""
        ##self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.npc_stmn), len(self.opp_stmn), len(self.actions), len(self.opponent_actions), len(self.opponent_actions), len(self.actions)))
        # print(q_table2.ndim)
        # print(q_table2.size)
        # print(q_table2.shape)
        #self.q_table2[:, :, :2, :, :, :, :, :, 1:9] = 0  # !Magdi!#
        #self.q_table2[:, :, 0:3, :, :, :, :, :, 0] = 5
        #self.q_table2[:, :, 4, :, :, :, :, :, 4:9] = 5
        #self.q_table2[:, :, 3, :, :, :, :, :, 4:9] = 0.5
        #self.q_table2[:, :, 2, :, :, :, :, :, 4:9] = 0.25
        ##self.q_table2[:, :, :, 0:1, :, :, :, :, 1:4] = -float('inf')
        ##self.q_table2[:, :, :, 0:1, :, :, :, :, 4] = -float('inf')
        ##self.q_table2[:, :, :, 0:2, :, :, :, :, 5] = -float('inf')
        ##self.q_table2[:, :, :, 0:3, :, :, :, :, 6] = -float('inf')
        ##self.q_table2[:, :, :, 0:4, :, :, :, :, 7] = -float('inf')
        # ue.print_string("Q_Table Class : Constructor")

    def create_table(self, name):
        ue.log("Creating Table")
        #self.q_table2 = np.zeros((len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.npc_stmn), len(self.opponent_actions), len(self.opponent_actions), len(self.opponent_actions), len(self.actions), len(self.actions), len(self.actions)), dtype='float16')
        # Generating random table manually to make it a float16 table instead of float64
        # for d1 in range(len(npc_hps)):
        #    for d2 in range(len(opp_hps)):
        #        for d3 in range(len(distances)):
        #            for d4 in range(len(npc_stmn)):
        #                for d5 in range(len(opponent_actions)):
        #                    for d6 in range(len(opponent_actions)):
        #                        for d7 in range(len(opponent_actions)):
        #                            for d8 in range(len(actions)):
        #                                for d9 in range(len(actions)):
        #                                    for d10 in range(len(actions)):
        #                                        t[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10] = np.float16(np.round(np.random.uniform(0, 5), 2))
        self.name = name
        self.q_table2 = np.round(np.random.uniform(low=-5, high=5, size=(len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.npc_stmn), len(self.opponent_actions), len(self.opponent_actions), len(self.opponent_actions), len(self.actions), len(self.actions), len(self.actions))).astype('float16'), 4)
        # print(q_table2.ndim)
        # print(q_table2.size)
        # print(q_table2.shape)
        #self.q_table2[:, :, :2, :, :, :, :, :, 1:9] = 0  # !Magdi!#
        #self.q_table2[:, :, 0:3, :, :, :, :, :, 0] = 5
        #self.q_table2[:, :, 4, :, :, :, :, :, 4:9] = 5
        #self.q_table2[:, :, 3, :, :, :, :, :, 4:9] = 0.5
        #self.q_table2[:, :, 2, :, :, :, :, :, 4:9] = 0.25
        self.q_table2[:, :, :, 0:1, :, :, :, :, 1:4] = -float('inf')
        self.q_table2[:, :, :, 0:2, :, :, :, :, 4] = -float('inf')
        self.q_table2[:, :, :, 0:3, :, :, :, :, 5] = -float('inf')
        self.q_table2[:, :, :, 0:4, :, :, :, :, 6] = -float('inf')
        self.q_table2[:, :, :, 0:3, :, :, :, :, 7] = -float('inf')

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
        # self.opp_ce_actions.append(L[6])
        if L[6] == "true":
            self.is_attacking = True
        else:
            self.is_attacking = False
        npc_c_stamina = int(L[7])
        old_npc_c_stamina = int(L[8])
        opp_c_stamina = int(L[9])
        old_opp_c_stamina = int(L[10])
        self.take_opponent_actions(L[11])
        ue.print_string(f"current_distance = {curr_dist}")
        if curr_npc_hp <= 0:
            curr_npc_hp = 0
        if curr_opp_hp <= 0:
            curr_opp_hp = 0
        if old_npc_hp <= 0:
            old_npc_hp = 0
        if old_opp_hp <= 0:
            old_opp_hp = 0

        if npc_c_stamina <= 0:
            npc_c_stamina = 0
        if opp_c_stamina <= 0:
            opp_c_stamina = 0
        if old_npc_c_stamina <= 0:
            old_npc_c_stamina = 0
        if opp_c_stamina <= 0:
            old_opp_c_stamina = 0
        state = (curr_npc_hp, curr_opp_hp, curr_dist, old_npc_hp, old_opp_hp, old_dist)

        old_distance_index = 0
        if state[5] > 700:
            old_distance_index = 0
        elif 700 >= state[2] > 550:
            old_distance_index = 1
        elif 550 >= state[2] > 300:
            old_distance_index = 2
        elif 300 >= state[2] > 0:
            old_distance_index = 3  # a&a

        new_distance_index = 0
        if state[5] > 700:
            new_distance_index = 0
        elif 700 >= state[2] > 550:
            new_distance_index = 1
        elif 550 >= state[2] > 300:
            new_distance_index = 2
        elif 300 >= state[2] > 0:
            new_distance_index = 3  # a&a

        new_NPC_HPi = 0
        if curr_npc_hp <= 25:
            new_NPC_HPi = 0
        elif 50 >= curr_npc_hp > 25:
            new_NPC_HPi = 1
        elif 75 >= curr_npc_hp > 50:
            new_NPC_HPi = 2
        elif 100 >= curr_npc_hp > 75:
            new_NPC_HPi = 3

        old_NPC_HPi = 0
        if old_npc_hp <= 25:
            old_NPC_HPi = 0
        elif 50 >= old_npc_hp > 25:
            old_NPC_HPi = 1
        elif 75 >= old_npc_hp > 50:
            old_NPC_HPi = 2
        elif 100 >= old_npc_hp > 75:
            old_NPC_HPi = 3

        new_opp_HPi = 0
        if curr_opp_hp <= 25:
            new_opp_HPi = 0
        elif 50 >= curr_opp_hp > 25:
            new_opp_HPi = 1
        elif 75 >= curr_opp_hp > 50:
            new_opp_HPi = 2
        elif 100 >= curr_opp_hp > 75:
            new_opp_HPi = 3

        old_opp_HPi = 0
        if old_opp_hp <= 25:
            old_opp_HPi = 0
        elif 50 >= old_opp_hp > 25:
            old_opp_HPi = 1
        elif 75 >= old_opp_hp > 50:
            old_opp_HPi = 2
        elif 100 >= old_opp_hp > 75:
            old_opp_HPi = 3

        old_state = (old_NPC_HPi, old_opp_HPi, old_distance_index, self.npc_stmn[old_npc_c_stamina], self.opp_ce_actions[0], self.opp_ce_actions[1], self.opp_ce_actions[2], self.NPC_ce_actions[1], self.NPC_ce_actions[2])
        new_state = (new_NPC_HPi, new_opp_HPi, new_distance_index, self.npc_stmn[npc_c_stamina], self.opp_ce_actions[1], self.opp_ce_actions[2], self.opp_ce_actions[3], self.NPC_ce_actions[2], self.NPC_ce_actions[3])
        ue.print_string(f"Old State {old_state}, New State {new_state}")

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state)

        return action

    def take_action(self, index_state):

        if np.random.random() > self.epsilon:
            ue.print_string("Take Max Q_Value")
            action = np.argmax(self.q_table2[index_state])
        else:
            ue.print_string("Explore: Random Action")
            action = np.random.randint(0, 9)

        self.NPC_ce_actions = self.NPC_ce_actions[1:4]
        self.NPC_ce_actions.append(action)
        ue.log(f"npc actions : {self.NPC_ce_actions}")
        return action

    def next_iterator_epsilon(self, name):
        self.iterator += 1
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        self.winning_rate.append((self.NPC_wins / (self.iterator) * 100))
        self.move_to_q.append(np.max(self.q_table2[0, :, :, :, :, :, :, :, :, 0]))
        self.attack_q.append(np.min(self.q_table2[0, :, :, 2:11, :, :, :, :, :, 4]))
        self.dodge_q.append(np.min(self.q_table2[0, :, :, 1:11, :, :, :, :, :, 3]))
        self.d_dealt.append(self.dmg_dealt)
        self.d_taken.append(self.dmg_taken)
        ue.log(
            f"damage dealt list = {self.d_dealt} \n damage taken list = {self.d_taken} \n move to q value list = {self.move_to_q} \n attack q value list = {self.attack_q} \n dodge q value list = {self.dodge_q}")
        self.dmg_dealt = 0
        self.dmg_taken = 0
        self.moves_counter = 0
        self.opp_ce_actions = [9, 9, 9, 9]
        self.NPC_ce_actions = [9, 9, 9, 9]
        # ue.print_string(f"MOVES ARE ZEROED YO!!!! {self.moves_counter}")
        if self.iterator % self.decay_every == 0 and self.iterator >= self.decay_from:
            self.epsilon *= self.eps_decay
            ue.print_string("DECAY")
    #    self.save_table(name)

    def final_damage(self, d_lsit):
        dl = d_list.split(',')
        d_tkn = dl[0] - dl[1]
        d_dlt = dl[2] - dl[3]
        self.d_dealt[-1] += d_dlt
        self.d_taken[-1] += d_tkn

    def calc_reward(self, current_state, old_state):  # fe h5a
        if old_state[1] * 25 - current_state[1] * 25 == 0:
            self.moves_counter += 1
        else:
            self.dmg_dealt += old_state[1] * 25 - current_state[1] * 25
        if old_state[0] * 25 - current_state[0] * 25 != 0:
            self.dmg_taken += old_state[0] * 25 - current_state[0] * 25

        action = self.NPC_ce_actions[3]
        succ_dodge = 0
        #ue.print_string(f"moves counter =  {self.moves_counter}")
        if self.is_attacking is True and (old_state[2] == 3 or old_state[2] == 2) and (action == 1 or action == 2 or action == 3 or action == 8) and \
                old_state[0] * 25 - current_state[0] * 25 == 0:
            succ_dodge = 5
            ue.print_string(f"successful dodge , with action {action},#moves {self.moves_counter}")
            ue.log(f"successful dodge , with action {action},#moves {self.moves_counter}")
        if action == 0:
            reward = (old_state[1] * 25 - current_state[1] * 25) - (old_state[0] * 25 - current_state[0] * 25)
        else:
            reward = (old_state[1] * 25 - current_state[1] * 25) - (old_state[0] * 25 - current_state[0] * 25) - (self.moves_counter * 0.22) + succ_dodge

        ue.print_string(f"Reward: {reward}, with action {action} ,#moves {self.moves_counter}")

        if old_state[1] * 25 - current_state[1] * 25 != 0:
            self.moves_counter = 0

        old_state_max_q = np.max(self.q_table2[old_state])
        current_q = self.q_table2[old_state][action]
        new_q = round((1 - self.learn_rate) * current_q + self.learn_rate * (reward + self.discount * old_state_max_q), 2)
        ue.log(f"Current_q {round(current_q, 4)} in state {old_state} and action {action} => New_q {round(new_q, 4)} in state {current_state}")
        ue.print_string(f"Current_q {round(current_q, 4)} in state {old_state} and action {action} => New_q {round(new_q, 4)} in state {current_state}")
        self.q_table2[old_state][action] = new_q

    def take_opponent_actions(self, actions):
        past_actions = actions.split('-')
        self.opp_ce_actions[0] = int(past_actions[0])
        self.opp_ce_actions[1] = int(past_actions[1])
        self.opp_ce_actions[2] = int(past_actions[2])
        self.opp_ce_actions[3] = int(past_actions[3])
        if self.opp_ce_actions[2] == 4 or self.opp_ce_actions[2] == 5 or self.opp_ce_actions[2] == 6 or self.opp_ce_actions[2] == 7:
            self.is_attacking = True
        else:
            self.is_attacking = False
        ue.print_string(f"opponent current action {past_actions[3]} , is attacking is {self.is_attacking} with actions : {self.opp_ce_actions}")
        ue.log(f"opponent actions {self.opp_ce_actions}")

    def save_table(self, name):
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        if self.iterator != 0:
            ue.log(f"winning_rate = {self.winning_rate}")
            filename = rf'./Q_Table{args[0]}.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table2, f)
                ue.print_string(f"{self.iterator} : Saved Q_Table")
            es = (self.iterator, self.epsilon, self.NPC_wins, self.opp_wins, self.winning_rate, self.d_taken, self.d_dealt, self.move_to_q, self.attack_q, self.dodge_q)
            filename = rf'./Episode{args[0]}.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(es, f)
                ue.log(f"{self.iterator} : Saved Episode , winning rate : {(self.NPC_wins / self.iterator) * 100}")

    def load_table(self, name):
        self.name = name
        try:
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
                self.winning_rate = es[4]
                self.d_taken = es[5]
                self.d_dealt = es[6]
                self.attack_q = es[7]
                self.dodge_q = es[8]
                ue.print_string(f"{self.iterator} : #Episodes is loaded")
                #ue.log(f"{self.q_table2[:, :, :, 4]}")
                str = f"{self.iterator},{self.NPC_wins},{self.opp_wins}"
                return str
        except:
            self.create_table(name)

    def __del__(self):
        try:
            self.save_table(f"{self.name},{self.NPC_wins},{self.opp_wins}")
            del self.q_table2
            ue.log("destructooooooooooooooooooooooooooooooooooooooooooor Table Deleted")
        except:
            ue.log("Destructor is failed")
