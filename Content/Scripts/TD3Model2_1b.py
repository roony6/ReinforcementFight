import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
import numpy as np
import unreal_engine as ue
import pickle

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.softmax(self.layer_3(x))
        return x



class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
         # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Building the whole Training Process into a class

class TD3(object):

    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.it = 0

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        actions_w = self.actor(state).cpu().data.numpy().flatten()
        ue.log(f'actions_weight = {actions_w}')
        return actions_w

    def train(self, s, discount=0.95, tau=0.005, policy_freq=2):

        # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
        batch_states, batch_next_states, batch_actions, batch_rewards = s
        state = torch.Tensor(batch_states).to(device)
        next_state = torch.Tensor(batch_next_states).to(device)
        action = torch.Tensor(batch_actions).to(device)
        reward = torch.Tensor(batch_rewards).to(device)
        # Step 5: From the next state s’, the Actor target plays the next action a’
        next_action = self.actor_target(next_state)

        # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
        '''noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)'''
        # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        ue.log(f'target_Q1 = {target_Q1} target_Q2 = {target_Q2}')
        # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
        target_Q = torch.min(target_Q1, target_Q2)
        ue.log(f'target_Q = {target_Q} \n reward = {reward}')
        # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
        target_Q = reward + (discount * target_Q).detach()
        ue.log(f'target_Q = {target_Q}')
        # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
        current_Q1, current_Q2 = self.critic(state, action)
        ue.log(f'current_Q1 = {current_Q1} current_Q2 = {current_Q2} target_Q = {target_Q}')
        # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
        if self.it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        self.it+=1



    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))



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
        self.d_taken = []
        self.dmg_taken = 0
        self.d_dealt = []
        self.dmg_dealt = 0
        self.move_to_q = []
        self.attack_q = []
        self.dodge_q = []
        self.opp_ce_actions = [9, 9, 9, 9]
        self.NPC_ce_actions = [9, 9, 9, 9]
        # hit_reward = 20
        # hit_penalty = -20
        # move_penalty = -2
        self.episodes = 100
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
        self.NPChp = 100
        self.OPPhp = 100
        self.oNPChp = 100
        self.oOPPhp = 100
        self.policy = TD3(8, 10)
        self.actions_prop = [0,0,0,0,0,0,0,0,0,0]
        self.buffer = ReplayBuffer()
        self.actions_batch=[]
        self.state_batch = []
        self.next_state_batch = []
        self.reward_batch = []

    def create_table(self, name):
        self.policy = TD3(8, 10)

    def intialize_states(self, cur_old_e_o_hp_dist):
        ue.print_string(f"Iterator :=> {self.iterator}, Epsilon :=> {self.epsilon}")
        if self.iterator == self.episodes:
            return -1
        L = cur_old_e_o_hp_dist.split(',')
        curr_npc_hp = int(L[0])
        curr_opp_hp = int(L[1])
        self.NPChp = curr_npc_hp
        self.OPPhp = curr_opp_hp
        curr_dist = int(L[2])
        old_npc_hp = int(L[3])
        old_opp_hp = int(L[4])
        self.oNPChp = old_npc_hp
        self.oOPPhp = old_opp_hp
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

        old_state = [old_npc_hp, old_opp_hp, old_dist, old_npc_c_stamina, old_opp_c_stamina, self.NPC_ce_actions[2], self.opp_ce_actions[1], self.opp_ce_actions[2]]
        new_state = [curr_npc_hp, curr_opp_hp, curr_dist, npc_c_stamina, opp_c_stamina, self.NPC_ce_actions[3], self.opp_ce_actions[2], self.opp_ce_actions[3]]
        #ue.print_string(f"Old State {old_state}, New State {new_state}")

        self.calc_reward(new_state, old_state)
        action = self.take_action(new_state)

        return action

    def take_action(self, index_state):
        for param in self.policy.actor.parameters():
            ue.log(f"wieghts :{param}")
            break

        self.actions_prop = self.policy.select_action(np.array(index_state))
        if self.iterator > 0.1 * self.episodes:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=self.actions_prop)
            ue.log(f"returned action :{action}")
        else:
            ue.print_string("Explore: Random Action")
            action = np.random.randint(0, 9)
        self.NPC_ce_actions = self.NPC_ce_actions[1:4]
        self.NPC_ce_actions.append(action)
        ue.log(f"npc actions :{self.NPC_ce_actions}")

        return action


    def next_iterator_epsilon(self, name):
        self.iterator += 1
        args = name.split(',')
        self.NPC_wins = int(args[1])
        self.opp_wins = int(args[2])
        self.winning_rate.append((self.NPC_wins / (self.iterator) * 100))
        #self.move_to_q.append(np.max(self.q_table2[0, :, :, :, :, :, :, :, 0]))
        #self.attack_q.append(np.min(self.q_table2[0, :, :, :, :, :, :, :, 4]))
        #self.dodge_q.append(np.min(self.q_table2[0, :, :, :, :, :, :, :, 3]))
        ue.log(f"dmg_dealt = {self.dmg_dealt} , dmg_taken = {self.dmg_taken}")
        self.d_dealt.append(self.dmg_dealt)
        self.d_taken.append(self.dmg_taken)
        ue.log(f"damage dealt list = {self.d_dealt} \n damage taken list = {self.d_taken} \n move to q value list = {self.move_to_q}")
        self.dmg_dealt = 0
        self.dmg_taken = 0
        self.moves_counter = 0
        self.opp_ce_actions = [9, 9, 9, 9]
        self.NPC_ce_actions = [9, 9, 9, 9]
        # ue.print_string(f"MOVES ARE ZEROED YO!!!! {self.moves_counter}")
        if self.iterator % self.decay_every == 0 and self.iterator >= self.decay_from:
            self.epsilon *= self.eps_decay
            ue.print_string("DECAY")

        # self.save_table(name)

    #def final_damage(self, d_list):
    #    dl = d_list.split(',')
    #    d_tkn = int(dl[0]) - int(dl[1])
    #    d_dlt = int(dl[2]) - int(dl[3])
    #    ue.log(f"damage taken = {dl[0]} - {dl[1]} = {d_tkn} \n damage dealt = {dl[2]} - {dl[3]} = {d_dlt}")
    #    self.d_dealt.append(d_dlt)
    #    self.d_taken.append(d_tkn)

    def calc_reward(self, current_state, old_state):  # fe h5a

        if self.oOPPhp - self.OPPhp == 0:
            self.moves_counter += 1
        else:
            self.dmg_dealt += self.oOPPhp - self.OPPhp

        if self.oNPChp - self.NPChp != 0:
            self.dmg_taken += self.oNPChp - self.NPChp

        action = self.NPC_ce_actions[3]
        succ_dodge = 0
        #ue.print_string(f"moves counter =  {self.moves_counter}")
        if self.is_attacking is True and (550 <= old_state[2] >= 0) and (action == 1 or action == 2 or action == 3 or action == 8) and \
                self.oNPChp - self.NPChp == 0:
            succ_dodge = 5
            ue.print_string(f"successful dodge , with action {action},#moves {self.moves_counter}")
            ue.log(f"successful dodge , with action {action},#moves {self.moves_counter}")

        reward = (self.oOPPhp - self.OPPhp) - (self.oNPChp - self.NPChp) - (self.moves_counter * 0.22) + succ_dodge
        ue.print_string(f"Reward: {reward}, with action {action} ,#moves {self.moves_counter}")

        if self.oOPPhp - self.OPPhp != 0:
            self.moves_counter = 0

        oss=[]
        oss.append(np.array(old_state, copy=False))
        css=[]
        css.append(np.array(current_state, copy=False))
        rs=[]
        rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        rewards[action] = reward
        rs.append(np.array(rewards, copy=False))
        acs=[]
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        actions[action] = 1
        ue.log(f"actions prop sent = {self.actions_prop}")
        if self.iterator > 0.1 * self.episodes:
            acs.append(np.array(self.actions_prop, copy=False))
        else:
            acs.append(np.array(actions, copy=False))

        s = [np.array(oss),np.array(css),np.array(acs),np.array([reward]).reshape(-1, 1)]
        self.policy.train(s)


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
            ue.log(f"self.winning_rate = {self.winning_rate}")
            self.policy.save("TD3_m2", rf'C:/Users/mosta/Documents')
            ue.print_string(f"{self.iterator} : Saved Q_Table")
            es = (
            self.iterator, self.epsilon, self.NPC_wins, self.opp_wins, self.winning_rate, self.d_taken, self.d_dealt,
            self.move_to_q, self.attack_q, self.dodge_q)
            filename = rf'./TD3_m2_Episode.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(es, f)
                ue.log(f"{self.iterator} : Saved Episode , winning rate : {(self.NPC_wins / self.iterator) * 100}")

    def load_table(self, name):
        self.name = name
        self.policy.load("TD3_m2", rf'C:/Users/mosta/Documents')
        try:

            ue.print_string(f"{self.iterator} : Q_Table is loaded")
            filename = rf'./TD3_m2_Episode.pickle'
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
                # ue.log(f"{self.q_table2[:, :, :, 4]}")
                str = f"{self.iterator},{self.NPC_wins},{self.opp_wins}"
            return str
        except:
            #self.create_table(name)
            ue.log("oooooooops")

    def __del__(self):
        self.save_table(f"{self.name},{self.NPC_wins},{self.opp_wins}")
        try:

            #del self.q_table2
            ue.log("destructooooooooooooooooooooooooooooooooooooooooooor Table Deleted")
        except:
            ue.log("Destructor failed")
