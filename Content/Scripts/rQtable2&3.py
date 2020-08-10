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
        self.winning_rate = []
        self.opp_ce_actions = [9, 9, 9, 9]
        self.NPC_ce_actions = [9, 9, 9, 9]
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

    def take_action(self):
        action = np.random.randint(0, 9)
        return action
