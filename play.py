from game  import SnakeGameAI
from agent import Agent

def watch():
    agent         = Agent(load_checkpoint=True)
    agent.epsilon = 0.0
    game          = SnakeGameAI(render=True)
    print("Watching Trained Bot - Ctrl+C to quit")
    episode = 0
    while True:
        state   = game.reset()
        episode += 1
        while True:
            action = agent.get_action(state)
            state, _, done, score = game.step(action)
            if done:
                break
        print("Game " + str(episode) + " Score: " + str(score))

if __name__ == "__main__":
    watch()
