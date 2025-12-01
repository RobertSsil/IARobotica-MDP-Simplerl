from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# Removido importação que causava erro

# IMPORTAÇÕES ATUALIZADAS
from StaticWarehouseMDP import StaticWarehouseMDP, GOAL_POS, START_POS, WIDTH, HEIGHT, OBSTACLE_POS
from simple_rl.mdp.StateClass import State 

def train_agent(mdp, episodes=5000):
    """ Treina o Agente Q-Learning. """
    print("Iniciando Treinamento do Agente Q-Learning...")
    q_agent = QLearningAgent(actions=mdp.get_actions(), name="Q-Robo-Estatico")
    
    run_agents_on_mdp([q_agent], mdp, 
                                instances=1, 
                                episodes=episodes, 
                                steps=200, 
                                open_plot=False)
    
    print("--- Treinamento Concluído! ---")
    return q_agent

def generate_learning_curve(mdp, num_instances=50, num_episodes=100):
    
    print("\nIniciando geração da Curva de Aprendizado (Tentativa de Plotagem Automática)...")
    q_agent = QLearningAgent(actions=mdp.get_actions(), name="Q-learning")
    rand_agent = RandomAgent(actions=mdp.get_actions(), name="Random")
    
    results_dir = run_agents_on_mdp(
                        agents=[q_agent, rand_agent], 
                        mdp=mdp, 
                        instances=num_instances, 
                        episodes=num_episodes, 
                        steps=200, 
                        open_plot=True 
                    )
    
    print(f"--- Curva de Aprendizado Concluída. O caminho salvo é: {results_dir} ---")

def animate_path(mdp, agent, num_steps=100):
    """ Simula o robô usando a política aprendida e cria uma animação. """
    print("\nIniciando Simulação (Animação)...")
    
    state = mdp.get_init_state()
    path_data = [] 
    total_reward = 0
    
    for t in range(num_steps):
        
        path_data.append(state.data) 
        
        if mdp.is_goal_state(state):
            print(f"Objetivo alcançado no passo {t}!")
            break

        action = agent.act(state, reward=0, learning=False)
        next_state = mdp.get_next_state(state, action)
        reward = mdp.get_reward(state, action, next_state)
        total_reward += reward
        
        state = next_state
    
    # 2. Configuração do Gráfico Inicial
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(-0.5, HEIGHT - 0.5)
    ax.set_xticks(np.arange(-0.5, WIDTH, 1))
    ax.set_yticks(np.arange(-0.5, HEIGHT, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='-', linewidth=2)
    plt.gca().invert_yaxis() 

    # MARCAR OBSTÁCULOS FIXOS
    obs_x = [o[0] for o in OBSTACLE_POS]
    obs_y = [o[1] for o in OBSTACLE_POS]
    ax.plot(obs_x, obs_y, 'ks', markersize=15, label='Obstáculo Fixo')
    
    # Marca Início e Fim (Estático)
    ax.plot(START_POS[0], START_POS[1], 'go', label='Início', markersize=12)
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'rP', markerfacecolor='red', markeredgecolor='black', label='Objetivo', markersize=15)
    
    # Cria o objeto móvel do Robô
    line_rob, = ax.plot([], [], 'bo', markersize=15, label='Robô (Agente)')
    
    ax.legend(loc='lower left')

    # 3. Função de Atualização da Animação
    def update(frame):
        """ Chamada a cada frame: atualiza a posição dos objetos. """
        
        rx, ry = path_data[frame] 
        
        line_rob.set_data([rx], [ry])
        
        ax.set_title(f"Passo {frame} - Robô: ({rx}, {ry})")
        
        return line_rob,

    # 4. Criação e Execução da Animação
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(path_data), 
        interval=500,
        blit=True,     
        repeat=False   
    )

    print(f"Recompensa Total na Simulação: {total_reward:.2f}")
    plt.show() 

if __name__ == '__main__':
    
    warehouse_mdp = StaticWarehouseMDP()
    
    trained_agent = train_agent(warehouse_mdp, episodes=5000)
    
    animate_path(warehouse_mdp, trained_agent, num_steps=100)
    
    generate_learning_curve(warehouse_mdp, num_instances=50, num_episodes=100)