import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        return self.net(x)
    def save(self, path="models/model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print("Model saved to " + path)
    def load(self, path="models/model.pth"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.eval()
            print("Model loaded from " + path)
        else:
            print("No checkpoint found, starting fresh.")

class QTrainer:
    def __init__(self, model, lr=1e-3, gamma=0.9):
        self.model     = model
        self.gamma     = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def train_step(self, states, actions, rewards, next_states, dones):
        import numpy as np
        states      = torch.tensor(np.array(states),      dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions     = torch.tensor(np.array(actions),     dtype=torch.long).to(self.device)
        rewards     = torch.tensor(np.array(rewards),     dtype=torch.float32).to(self.device)
        dones       = torch.tensor(np.array(dones),       dtype=torch.bool).to(self.device)
        if states.dim() == 1:
            states      = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions     = actions.unsqueeze(0)
            rewards     = rewards.unsqueeze(0)
            dones       = dones.unsqueeze(0)
        q_pred = self.model(states)
        with torch.no_grad():
            q_next   = self.model(next_states)
            q_target = q_pred.clone()
            for i in range(len(dones)):
                new_q = rewards[i]
                if not dones[i]:
                    new_q = rewards[i] + self.gamma * torch.max(q_next[i])
                q_target[i][actions[i]] = new_q
        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
