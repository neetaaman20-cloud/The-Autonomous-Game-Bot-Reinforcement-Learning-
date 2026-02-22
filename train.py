import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from game   import SnakeGameAI
from agent  import Agent
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--no-render", action="store_true")
parser.add_argument("--load",      action="store_true")
parser.add_argument("--episodes",  type=int, default=500)
args = parser.parse_args()

RENDER   = not args.no_render
EPISODES = args.episodes

def plot_scores(scores, mean_scores, path="logs/training_curve.png"):
    plt.figure(figsize=(10, 4))
    plt.style.use("dark_background")
    plt.plot(scores,      color="#00E6B4", alpha=0.4, label="Score")
    plt.plot(mean_scores, color="#50A0FF", linewidth=2, label="Mean Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("DQN Snake Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#0A0C14")
    plt.close()

def train():
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    agent  = Agent(load_checkpoint=args.load)
    game   = SnakeGameAI(render=RENDER)
    logger = Logger()
    scores, mean_scores = [], []
    total_score = 0
    best_score  = 0
    print("Autonomous RL Snake Bot - Training Started")
    for episode in range(1, EPISODES + 1):
        state          = game.reset()
        episode_reward = 0
        while True:
            action                          = agent.get_action(state)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            state          = next_state
            episode_reward += reward
            if done:
                break
        agent.train_long_memory()
        agent.decay_epsilon()
        agent.n_games += 1
        if score > best_score:
            best_score = score
            agent.save()
        total_score += score
        mean_score   = total_score / episode
        scores.append(score)
        mean_scores.append(mean_score)
        logger.log(episode, score, mean_score, agent.epsilon, episode_reward)
        if episode % 10 == 0:
            plot_scores(scores, mean_scores)
            print("Ep " + str(episode) + " Score " + str(score) + " Best " + str(best_score) + " Mean " + str(round(mean_score, 2)))
    print("Done. Best score: " + str(best_score))
    plot_scores(scores, mean_scores)

if __name__ == "__main__":
    train()
