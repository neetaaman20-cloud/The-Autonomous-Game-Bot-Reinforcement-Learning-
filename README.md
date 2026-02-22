# 🤖 The Autonomous Game Bot (Reinforcement Learning)

A Deep Q-Network (DQN) AI agent that teaches itself to play Snake from scratch — no human input, no hardcoded rules. Just a neural network, rewards, and thousands of games.

Built with **Python**, **PyTorch**, and **Pygame**.

---

## 📸 Preview

> The bot starts completely random. After ~300 games it plays intelligently, avoiding walls and chasing food efficiently.

---

## 🧠 How It Works

The bot uses **Deep Q-Learning (DQN)** — a reinforcement learning algorithm where the agent:

1. Observes the current game state (11 inputs)
2. Decides an action: go straight, turn left, or turn right
3. Receives a reward (+10 for food, -10 for dying)
4. Learns from experience using a neural network
5. Gets better every single game

| Component | Details |
|-----------|---------|
| Algorithm | Deep Q-Network (DQN) |
| Framework | PyTorch |
| Graphics  | Pygame |
| State     | 11 boolean values |
| Actions   | Straight / Turn Right / Turn Left |
| Rewards   | +10 food, -10 death |
| Memory    | Experience replay (100,000 buffer) |
| Network   | 11 → 256 → 256 → 3 (ReLU activations) |
| Exploration | Epsilon-greedy with exponential decay |

---

## 📁 Project Structure

```
The-Autonomous-Game-Bot-Reinforcement-Learning/
├── game.py            # Snake game engine + Pygame graphics
├── agent.py           # DQN agent (memory, epsilon-greedy, actions)
├── model.py           # Neural network (DQN) + QTrainer
├── train.py           # Main training loop — run this to train
├── play.py            # Watch the trained bot play
├── logger.py          # Saves training stats to CSV
├── config.py          # All hyperparameters in one place
├── requirements.txt   # Python dependencies
├── models/            # Auto-created: saved model checkpoints
└── logs/              # Auto-created: training curves + CSV logs
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AnugrahSingh/The-Autonomous-Game-Bot-Reinforcement-Learning.git
cd The-Autonomous-Game-Bot-Reinforcement-Learning
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the bot
```bash
python3 train.py
```

### 5. Train faster (no graphics window)
```bash
python3 train.py --no-render
```

### 6. Resume training from a saved checkpoint
```bash
python3 train.py --load
```

### 7. Watch the trained bot play
```bash
python3 play.py
```

---

## ⚙️ Hyperparameters

All tunable settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.001 | How fast the network learns |
| `GAMMA` | 0.9 | Discount factor for future rewards |
| `EPSILON_START` | 1.0 | Starting exploration rate |
| `EPSILON_END` | 0.01 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.995 | How fast exploration decreases |
| `BATCH_SIZE` | 1000 | Samples per training step |
| `MAX_MEMORY` | 100,000 | Replay buffer size |
| `HIDDEN_SIZE` | 256 | Neurons per hidden layer |

---

## 📊 Training Output

- **Live game window** showing the snake learning in real time
- **Terminal logs** every 10 episodes with score, best score, mean, epsilon
- **Training curve** saved to `logs/training_curve.png`
- **Best model** auto-saved to `models/model.pth`

---

## 📦 Requirements

```
pygame>=2.5.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## 💡 Tips

- Run `--no-render` for **10x faster** training
- After **200-300 episodes** the bot starts playing well
- After **500 episodes** it plays at a high level
- Increase `HIDDEN_SIZE` to `512` for more complex learning

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch |
| **Game Engine** | Pygame |
| **Data & Math** | NumPy |
| **Visualization** | Matplotlib |
| **Algorithm** | Deep Q-Network (DQN) |
| **Training** | Experience Replay + Epsilon-Greedy |
| **Model Format** | PyTorch `.pth` checkpoint |
| **Logging** | CSV + PNG training curves |
| **IDE** | VS Code |
| **Version Control** | Git + GitHub |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Author

**Anugrah Singh**
Built with ❤️ and reinforcement learning.

[![GitHub](https://img.shields.io/badge/GitHub-AnugrahSingh-181717?style=flat&logo=github)](https://github.com/AnugrahSingh)
