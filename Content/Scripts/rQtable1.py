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
        self.opp_ce_actions = []
        self.episodes = 2000
        self.learn_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.9
        self.eps_decay = 0.75
        self.decay_every = (5/100) * self.episodes
        self.decay_from = (10/100) * self.episodes
        self.iterator = 0
        self.moves_counter = 0
        self.is_attacking = False
        self.q_table2 = np.random.uniform(low=0, high=5, size=(len(self.npc_hps), len(self.opp_hps), len(self.distances), len(self.actions)))
        self.q_table2[:, :, :2, 4] = -100
        self.q_table2[:, :, 3, 4] = 5
        self.q_table2[:, :, 2, 4] = 1

    def take_action(self):
        action = np.random.randint(0, 6)
        return action