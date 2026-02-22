import random
import numpy as np
from collections import deque
from model import DQN, QTrainer

MAX_MEMORY    = 100000
BATCH_SIZE    = 1000
LR            = 1e-3
GAMMA         = 0.9
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995

class Agent:
    def __init__(self, load_checkpoint=False):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.memory  = deque(maxlen=MAX_MEMORY)
        self.model   = DQN(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        if load_checkpoint:
            self.model.load()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            batch = list(self.memory)
        else:
            batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(
            np.array(states), np.array(actions),
            np.array(rewards), np.array(next_states), np.array(dones),
        )
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        import torch
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state_t)
        return int(q_vals.argmax().item())
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    def save(self):
        self.model.save("models/model.pth")
    def load(self):
        self.model.load("models/model.pth")
