
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import random
from collections import defaultdict

# Import the MDP and constants
from StaticWarehouseMDP import StaticWarehouseMDP, GOAL_POS, START_POS, WIDTH, HEIGHT, OBSTACLE_POS

# Parâmetros de treino (ajustados)
EPISODES = 5000
MAX_STEPS = 200
ALPHA = 0.7
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995


class SimpleQLearningAgent:
    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPS_START, name="QAgent"):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = name
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def get_state_key(self, state):
        return tuple(state.data)

    def act(self, state, explore=True):
        s = self.get_state_key(state)
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.q[s]
        max_val = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done=False):
        s = self.get_state_key(state)
        sp = self.get_state_key(next_state)
        q_sa = self.q[s][action]
        max_next = max(self.q[sp].values()) if not done else 0.0
        self.q[s][action] = q_sa + self.alpha * (reward + self.gamma * max_next - q_sa)

    def set_epsilon(self, eps):
        self.epsilon = eps

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q), f)

    def load(self, path):
        with open(path, 'rb') as f:
            qdict = pickle.load(f)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions}, qdict)


def train_agent(mdp, episodes=EPISODES):
    agent = SimpleQLearningAgent(actions=mdp.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=EPS_START)
    returns = []

    for ep in range(1, episodes + 1):
        state = mdp.get_init_state()
        total_reward = 0
        for t in range(MAX_STEPS):
            action = agent.act(state, explore=True)
            next_state = mdp.get_next_state(state, action)
            reward = mdp.get_reward(state, action, next_state)
            done = mdp.is_goal_state(next_state)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        # decaimento epsilon
        agent.set_epsilon(max(EPS_END, agent.epsilon * EPS_DECAY))
        returns.append(total_reward)
        if ep % 50 == 0 or ep == 1:
            print(f"Episódio {ep}/{episodes} - Recompensa: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    print("--- Treinamento Concluído ---")
    return agent, returns


def evaluate_agent(mdp, agent, episodes=50):
    total = 0
    for _ in range(episodes):
        state = mdp.get_init_state()
        ep_reward = 0
        for t in range(MAX_STEPS):
            action = agent.act(state, explore=False)
            next_state = mdp.get_next_state(state, action)
            reward = mdp.get_reward(state, action, next_state)
            state = next_state
            ep_reward += reward
            if mdp.is_goal_state(state):
                break
        total += ep_reward
    return total / episodes


def print_q_table_segment(agent, mdp, n_states=6):
    print("\n========= Q-table (amostra) =========")
    # mostra estados ao redor do início
    s0 = mdp.get_init_state()
    neighbors = [s0]
    for a in mdp.get_actions():
        ns = mdp.get_next_state(s0, a)
        if ns.data != s0.data:
            neighbors.append(ns)
    neighbors = neighbors[:n_states]

    for s in neighbors:
        sk = tuple(s.data)
        print(f"Estado {sk}:")
        for a, v in agent.q[sk].items():
            print(f"  {a}: {v:.2f}")
    print("=====================================")


def animate_path(mdp, agent, max_steps=MAX_STEPS):
    state = mdp.get_init_state()
    path = [state.data]
    total_reward = 0
    for t in range(max_steps):
        action = agent.act(state, explore=False)
        next_state = mdp.get_next_state(state, action)
        reward = mdp.get_reward(state, action, next_state)
        total_reward += reward
        state = next_state
        path.append(state.data)
        if mdp.is_goal_state(state):
            print(f"Objetivo alcançado em {t+1} passos")
            break

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(-0.5, HEIGHT - 0.5)
    ax.set_xticks(np.arange(-0.5, WIDTH, 1))
    ax.set_yticks(np.arange(-0.5, HEIGHT, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='-', linewidth=1)
    plt.gca().invert_yaxis()

    obs_x = [o[0] for o in OBSTACLE_POS]
    obs_y = [o[1] for o in OBSTACLE_POS]
    ax.plot(obs_x, obs_y, 'ks', markersize=15, alpha=0.9)
    ax.plot(START_POS[0], START_POS[1], 'go', markersize=12)
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'rP', markerfacecolor='red', markeredgecolor='black', markersize=15)

    line_rob, = ax.plot([], [], 'bo', markersize=12)

    def update(i):
        x, y = path[i]
        line_rob.set_data([x], [y])
        ax.set_title(f"Passo {i} - Robô: ({x}, {y})")
        return line_rob,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=350, blit=True, repeat=False)
    print(f"Recompensa total na simulação: {total_reward:.2f}")
    plt.show()


def plot_learning_curve(returns):
    plt.figure(figsize=(8, 4))
    plt.plot(returns)
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa total por episódio')
    plt.title('Curva de Aprendizado')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    mdp = StaticWarehouseMDP()

    agent, returns = train_agent(mdp, episodes=EPISODES)
    agent.save('trained_q_table.pkl')

    #print_q_table_segment(agent, mdp)

    eval_mean = evaluate_agent(mdp, agent, episodes=50)
    print(f"Recompensa média (50 episódios de avaliação): {eval_mean:.2f}")

    animate_path(mdp, agent)
    plot_learning_curve(returns)

    # carregar o q_table com agent.load('trained_q_table.pkl')
