import numpy as np
import unreal_engine as ue
import pickle


class Game:
    def __init__(self):
        self.actions = {0: 'MoveToPlayer', 1: 'DodgeRight', 2: 'DodgeLeft', 3: 'DodgeBack', 4: 'Attack', 5: 'Idle'}
        self.npc_hps = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12, 65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.opp_hps = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7, 40: 8, 45: 9, 50: 10, 55: 11, 60: 12, 65: 13, 70: 14, 75: 15, 80: 16, 85: 17, 90: 18, 95: 19, 100: 20}
        self.distances = {0: '2000 - 800', 1: '800 - 500', 2: '500 - 200', 3: '200 - 0'}
        self.NPC_wins = 0
        self.opp_wins = 0
        self.winning_rate = []
        self.d_taken = []
        self.dmg_taken = 0
        self.d_dealt = []
        self.dmg_dealt = 0
        self.move_to_q = []
        self.attack_q = []
        self.dodge_q = []
        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.opp_ce_actions = []
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

    def create_table(self, name):
        ue.log("Creating Table")
        self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.actions)))
        self.q_table2[:, :, :2, 4] = -100
        self.q_table2[:, :, 3, 4] = 5
        self.q_table2[:, :, 2, 4] = 1
        self.name = name

    def intialize_states(self, cur_old_e_o_hp_dist):
        if self.iterator == self.episodes:
            return -1

        L = cur_old_e_o_hp_dist.split(',')
        curr_npc_hp = int(L[0])
        curr_opp_hp = int(L[1])
        curr_dist = int(L[2])
        old_npc_hp = int(L[3])
        old_opp_hp = int(L[4])
        old_dist = int(L[5])
        if L[6] == "false":
            self.is_attacking = False
        else:
            self.is_attacking = True
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
        if 2000 >= state[5] > 800:
            old_distance_index = 0
        elif 800 >= state[5] > 500:
            old_distance_index = 1
        elif 500 >= state[5] > 200:
            old_distance_index = 2
        elif 200 >= state[5] > 0:
            old_distance_index = 3

        new_distance_index = 0
        if 2000 >= state[2] > 800:
            new_distance_index = 0
        elif 800 >= state[2] > 500:
            new_distance_index = 1
        elif 500 >= state[2] > 200:
            new_distance_index = 2
        elif 200 >= state[2] > 0:
            new_distance_index = 3  # a&a

        old_state = (self.npc_hps[state[3]], self.opp_hps[state[4]], old_distance_index)
        new_state = (self.npc_hps[state[0]], self.opp_hps[state[1]], new_distance_index)
        # ue.print_string(f"Old State {old_state}, New State {new_state}")

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state[0], new_state[1], new_state[2])

        return action

    def take_action(self, p_health, e_health, d_stance):
        index_state = (p_health, e_health, d_stance)

        if np.random.random() > self.epsilon:
            # ue.print_string("Take Max Q_Value")
            action = np.argmax(self.q_table2[index_state])
            # ue.log(f"{self.q_table2[index_state]}")
        else:
            # ue.print_string("Explore: Random Action")
            action = np.random.randint(0, 6)
        return action

    def next_iterator_epsilon(self, name):
        ue.print_string(f"Iterator :=> {self.iterator}, Epsilon :=> {self.epsilon}")
        self.iterator += 1
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        self.winning_rate.append((self.NPC_wins / (self.iterator) * 100))
        self.move_to_q.append(np.max(self.q_table2[0, :, :, 0]))
        self.attack_q.append(np.min(self.q_table2[0, :, :, 4]))
        self.dodge_q.append(np.min(self.q_table2[0, :, :, 3]))
        ue.log(f"dmg_dealt = {self.dmg_dealt} , dmg_taken = {self.dmg_taken}")
        self.d_dealt.append(self.dmg_dealt)
        self.d_taken.append(self.dmg_taken)
        ue.log(
            f"damage dealt list = {self.d_dealt} \n damage taken list = {self.d_taken} \n move to q value list = {self.move_to_q}")
        self.dmg_dealt = 0
        self.dmg_taken = 0
        self.moves_counter = 0
        # ue.print_string(f"MOVES ARE ZEROED YO!!!! {self.moves_counter}")
        if self.iterator % self.decay_every == 0 and self.iterator >= self.decay_from:
            self.epsilon *= self.eps_decay
            ue.print_string("DECAY")

    #    self.save_table(name)

    #def final_damage(self, d_list):
    #    dl = d_list.split(',')
    #    d_tkn = int(dl[0]) - int(dl[1])
    #    d_dlt = int(dl[2]) - int(dl[3])
    #    ue.log(f"damage taken = {dl[0]} - {dl[1]} = {d_tkn} \n damage dealt = {dl[2]} - {dl[3]} = {d_dlt}")
    #    self.d_dealt.append(d_dlt)
    #   self.d_taken.append(d_tkn)

    def calc_reward(self, current_state, old_state):
        if old_state[1] * 5 - current_state[1] * 5 == 0:
            self.moves_counter += 1
        else:
            self.dmg_dealt += old_state[1] * 5 - current_state[1] * 5
        if old_state[0] * 5 - current_state[0] * 5 != 0:
            self.dmg_taken += old_state[0] * 5 - current_state[0] * 5

        action = self.take_action(old_state[0], old_state[1], old_state[2])
        succ_dodge = 0
        # ue.print_string(f"player is attacking is {self.is_attacking}")
        if self.is_attacking is True and old_state[2] == 3 and (action == 1 or action == 2 or action == 3) and old_state[0] * 5 - current_state[0] * 5 == 0:
            succ_dodge = 5
            # ue.print_string(f"successful dodge {self.is_attacking}")
            # ue.log(f"successful dodge , with action {action},#moves {self.moves_counter}")

        reward = (old_state[1] * 5 - current_state[1] * 5) - (old_state[0] * 5 - current_state[0] * 5) - (self.moves_counter * 0.22) + succ_dodge
        # ue.print_string(f"Reward: {reward}, with action {action}, #moves {self.moves_counter}")

        if old_state[1] * 5 - current_state[1] * 5 != 0:
            self.moves_counter = 0

        max_future_q = np.max(self.q_table2[current_state])
        current_q = self.q_table2[old_state][action]
        new_q = (1 - self.learn_rate) * current_q + self.learn_rate * (reward + self.discount * max_future_q)
        ue.log(f"Current_q {round(current_q, 2)} in state {old_state} and action {action} => New_q {round(new_q, 2) } in state {current_state}")
        self.q_table2[old_state][action] = new_q

    def save_table(self, name):
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        if self.iterator != 0:
            ue.log(f"iterartor = {self.iterator}")
            ue.log(f"self.winning_rate = {self.winning_rate}")
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
                itr_n_o_wins = f"{self.iterator},{self.NPC_wins},{self.opp_wins}"
                return itr_n_o_wins
        except:
            self.create_table(name)

    def __del__(self):
        try:
            self.save_table(f"{self.name},{self.NPC_wins},{self.opp_wins}")
            del self.q_table2
            ue.log("destructooooooooooooooooooooooooooooooooooooooooooor Table Deleted")
        except:
            ue.log("Destructor failed")
