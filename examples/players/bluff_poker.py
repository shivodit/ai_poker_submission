from pypokerengine.players import BasePokerPlayer
import random as rand

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from pypokerengine.utils.card_utils import Card, Deck
import random
from collections import namedtuple
import pickle

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())

def gen_card_im(card):
    a = np.zeros((4, 13))
    s = suits.index(card.suit)
    r = ranks.index(card.rank)
    a[s, r] = 1
    return np.pad(a, ((6, 7), (2, 2)), 'constant', constant_values=0)

streep_map = {
    'preflop': 0,
    'flop': 1,
    'turn': 2,
    'river': 3
}

def get_street(s):
    val = [0, 0, 0, 0]
    val[streep_map[s]] = 1
    return val

def process_img(img):
    return np.reshape(img, [17 * 17 * 1])

class ExperienceBuffer():
    def __init__(self, buffer_size=5_000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(self.buffer) + len(experience) - self.buffer_size] = []
        self.buffer.extend(experience)
    
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7])


def get_action_by_num(action_num, valid_actions, is_train=True):
    if action_num == 0:
        action, amount = valid_actions[0]['action'], valid_actions[0]['amount']
    elif action_num == 1:
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    elif action_num == 2:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']
    elif action_num == 3:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']
    elif action_num == 4:
        action, amount = valid_actions[2]['action'], int(valid_actions[2]['amount']['max'] // 2)
        
    if not is_train and amount == -1:
        print(action, amount)
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    return action, amount

def img_from_state(hole_card, round_state):
    imgs = np.zeros((8, 17, 17))
    for i, c in enumerate(hole_card):
        imgs[i] = gen_card_im(Card.from_str(c))

    for i, c in enumerate(round_state['community_card']):
        imgs[i + 2] = gen_card_im(Card.from_str(c))

    imgs[7] = imgs[:7].sum(axis=0)
#     return imgs
    return np.swapaxes(imgs, 0, 2)[:, :, -1:]


class DQNPlayer(nn.Module):
    def __init__(self, h_size=128, lr=0.0001, total_num_actions=5, is_double=True,
                 is_main=True, is_restore=True, is_train=False, debug=False):
        super(DQNPlayer, self).__init__()
        self.h_size = h_size
        self.lr = lr
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_restore = is_restore
        self.is_train = is_train
        self.debug = debug

        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, self.h_size, 5)

        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(self.h_size + 128, 256)
        self.fc4 = nn.Linear(256, self.h_size)

        if is_double:
            self.fc_A = nn.Linear(self.h_size // 2, total_num_actions)
            self.fc_V = nn.Linear(self.h_size // 2, 1)
        else:
            self.fc_out = nn.Linear(self.h_size, 5)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, features):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(x.size(0), -1)

        features = F.elu(self.fc1(features))
        features = F.elu(self.fc2(features))

        x = torch.cat((x, features), dim=1)
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))

        if self.is_double:
            A = self.fc_A(x)
            V = self.fc_V(x)
            return V + (A - A.mean())
        else:
            return self.fc_out(x)

    def choose_action(self, state, features):
        state = torch.from_numpy(state).float().unsqueeze(0)
        features = torch.from_numpy(np.array(features)).float().unsqueeze(0)
        probs = self.forward(state, np.array(features))
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def save_model(model):
    with open('model.pkl', 'wb') as f:
        torch.save(model.state_dict(), f)

class BluffPlayer(BasePokerPlayer):

  def __init__(self):
    self.model = DQNPlayer()
    weight_file = 'model.pkl'
    with open('hole_card_estimation.pkl', 'rb') as f:
        self.hole_card_est = pickle.load(f)

    try:
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
    except FileNotFoundError:
        print("Weight file not found. Using initial model.")

  def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
    ratio = [fold_ratio, call_ratio, raise_ratio]
    scaled_ratio = [ 1.0 * num / sum(ratio) for num in ratio]
    self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

  def declare_action(self, valid_actions, hole_card, round_state):
    street = round_state['street']
    bank = round_state['pot']['main']['amount']
    stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
    other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]
    dealer_btn = round_state['dealer_btn']
    small_blind_pos = round_state['small_blind_pos']
    big_blind_pos = round_state['big_blind_pos']
    next_player = round_state['next_player']
    round_count = round_state['round_count']
    estimation = self.hole_card_est[(hole_card[0], hole_card[1])]

    
    self.features = get_street(street)
    self.features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
    self.features.extend(other_stacks)
    self.features.append(estimation)
    
    img_state = img_from_state(hole_card, round_state)
    img_state = process_img(img_state)
    action,amount = self.model.choose_action(img_state, self.features)
    return action, amount

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    self.start_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]
    # estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, new_action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass


def train_model(model, buffer, batch_size, gamma):
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    optimizer = model.optimizer
    criterion = nn.MSELoss()

    if len(buffer.buffer) < batch_size:
        return

    transitions = buffer.sample(batch_size)
    batch = Experience(*zip(*transitions))

    state_batch = torch.from_numpy(np.array(batch.state)).float()
    action_batch = torch.from_numpy(np.array(batch.action)).long()
    reward_batch = torch.from_numpy(np.array(batch.reward)).float()
    next_state_batch = torch.from_numpy(np.array(batch.next_state)).float()
    done_batch = torch.from_numpy(np.array(batch.done)).float()

    state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = model(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) * (1 - done_batch) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn_model(model, buffer, batch_size=32, gamma=0.99, num_episodes=1000):
    for episode in range(num_episodes):

        # Train the model
        train_model(model, buffer, batch_size, gamma)

    # Save the model
    save_model(model)

# Create an instance of the DQNPlayer model
model = DQNPlayer()

# Create an instance of the ExperienceBuffer
buffer = ExperienceBuffer()

# Set the hyperparameters for training
batch_size = 32
gamma = 0.99
num_episodes = 1000



# Train the DQN model
train_dqn_model(model, buffer, batch_size, gamma, num_episodes)
