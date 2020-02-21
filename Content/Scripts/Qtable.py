import numpy as np
import itertools
import unreal_engine as ue

ue.print_string('Begin Play on Hero class')


class Fighter:

    # this is called on game start
    def begin_play(self):
        ue.print_string('Begin Play on Hero class')
        d = np.random.rand()
        ue.print_string(d)
        
    def __init__(self, x, y):
        self.health = 100
        self.x = x
        self.y = y
        self.distance = 4

    def action(self, choice):
        if choice == 0:
            self.dodge(x=-1, y=0)
        elif choice == 1:
            self.dodge(x=1, y=0)
        elif choice == 2:
            self.dodge(x=0, y=1)
        elif choice == 3:
            self.dodge(x=0, y=-1)
        elif choice == 4:
            self.dodge(x=1, y=1)
        elif choice == 5:
            self.dodge(x=-1, y=1)
        elif choice == 6:
            self.dodge(x=1, y=-1)
        elif choice == 7:
            self.dodge(x=-1, y=-1)
        elif choice == 8:
            self.attack()
        elif choice == 9:
            pass

    def dodge(self, x, y):
        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > 9:
            self.x = 9

        if self.y < 0:
            self.y = 0
        elif self.y > 9:
            self.y = 9

    @staticmethod
    def attack():
        power = 25
        return power

    def set_dist(self, dis):
        if dis > 9:
            self.distance = 9
        else:
            self.distance = dis


actions = {0: 'ML', 1: 'MR', 2: 'MF', 3: 'MB', 4: 'MFR', 5: 'MFL', 6: 'MBR', 7: 'MBL', 8: 'AT', 9: 'ID'}
play_health = {0: 0, 25: 1, 50: 2, 75: 3, 100: 4}
enmy_health = {0: 0, 25: 1, 50: 2, 75: 3, 100: 4}
x_coor = [i for i in range(10)]
y_coor = [i for i in range(10)]
distance = [i for i in range(10)]


episodes = 5000
hit_reward = 200
hit_penalty = -200
move_penalty = -2
learn_rate = 0.1
discount = 0.95

states = list(itertools.product(*[play_health.values(), enmy_health.values(), distance]))
# print(states)
# print(len(states))

q_table = np.random.uniform(low=2, high=0, size=(len(states), len(actions)))
q_table2 = np.random.uniform(low=0, high=5, size=(len(play_health), len(enmy_health), len(distance), len(actions)))
# print(q_table2.ndim)
# print(q_table2.shape)
# print(q_table2.size)
q_table2[:, :, 3:4, 8:10] = 0

enemy_wins = 0
player_wins = 0

for eps in range(episodes):
    player = Fighter(4, 6)
    enemy = Fighter(4, 2)
    eps_reward = 0
    who_won = ""
    while player.health > 0 or enemy.health > 0:
        # print(f"{player.health} player's health")
        # print(f"{enemy.health} enemy's health")
        reward = 0
        state = (play_health[player.health], enmy_health[enemy.health], int(player.distance))
        action = np.argmax(q_table2[state])
        # print(f"{action} player's action")
        player.action(action)
        # enemy_action = np.random.randint(0, 10)
        enemy_action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01])
        # enemy_action = np.random.randint(5, 10)
        enemy.action(enemy_action)

        dist = np.sqrt(pow((player.x - enemy.x), 2) + pow((player.y - enemy.y), 2))
        player.set_dist(dist)
        enemy.set_dist(dist)

        # if player.x == enemy.x - 1 or player.y == enemy.y - 1 or player.x - 1 == enemy.x or player.y - 1 == enemy.y:
        if player.distance and enemy.distance <= 4:
            # print("in fight state")
            if enemy_action == 8 and action == 8:
                enemy.health -= player.attack()
                player.health -= enemy.attack()
                reward = hit_reward * 2
            elif enemy_action != 8 and action == 8:
                enemy.health -= player.attack()
                reward = hit_reward
            elif enemy_action == 8 and action != 8:
                player.health -= enemy.attack()
                reward = hit_penalty
            else:
                reward = move_penalty
        # else:
            # print("not in fight state")
        new_state = (play_health[player.health], enmy_health[enemy.health], int(player.distance))
        max_future_q = np.max(q_table2[new_state])
        current_q = q_table2[new_state][action]

        if reward == hit_reward:
            new_q = reward
        elif reward == hit_penalty:
            new_q = reward
        else:
            new_q = (1 - learn_rate) * current_q + learn_rate * (reward + discount * max_future_q)

        q_table2[state][action] = new_q
        eps_reward += reward
        if player.health == 0:
            who_won = "Enemy"
            enemy_wins += 1
            break
        elif enemy.health == 0:
            who_won = "Player"
            player_wins += 1
            break
    print(f"{who_won}, won in episode {eps}")
    # break

print(f"Player Win rate {player_wins / episodes}")
print(f"Enemy Win rate {enemy_wins / episodes}")
