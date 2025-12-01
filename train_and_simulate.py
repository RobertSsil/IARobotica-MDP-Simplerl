from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_agents_on_mdp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Importa as definições do seu MDP
from DynamicWarehouseMDP import DynamicWarehouseMDP, GOAL_POS, START_POS, WIDTH, HEIGHT

def train_agent(mdp, episodes=5000):
    """ Treina o Agente Q-Learning. """
    print("Iniciando Treinamento do Agente Q-Learning (5000 episódios)...")
    q_agent = QLearningAgent(actions=mdp.get_actions(), name="Q-Robo-Dinamico")
    
    # Executa o treinamento
    run_agents_on_mdp([q_agent], mdp, 
                                instances=1, 
                                episodes=episodes, 
                                steps=200, 
                                open_plot=False)
    
    print("--- Treinamento Concluído! ---")
    return q_agent

def animate_path(mdp, agent, num_steps=50):
    """ Simula o robô usando a política aprendida e cria uma animação. """
    
    # 1. Coleta a Trajetória Passo a Passo
    state = mdp.get_init_state()
    path_data = [] # Armazena a tupla (rx, ry, ox, oy) a cada passo
    total_reward = 0
    
    for t in range(num_steps):
        
        # Coletamos a tupla de dados da classe de estado
        path_data.append(state.data) 
        
        if state.is_terminal(): # Usa o método is_terminal da classe DynamicState
            print(f"Objetivo alcançado no passo {t}!")
            break

        # Agente escolhe a Ação Ótima (max Q-value)
        action = agent.act(state, reward=0, learning=False)
        
        # Transição de estado
        next_state = mdp.get_next_state(state, action)
        
        # Acumula a recompensa
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

    # Marca Início e Fim (Estático)
    ax.plot(START_POS[0], START_POS[1], 'g^', label='Início', markersize=12)
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'P', markerfacecolor='gold', markeredgecolor='black', label='Objetivo', markersize=15)
    
    # Cria os objetos móveis
    line_rob, = ax.plot([], [], 'bo', markersize=15, label='Robô (Agente)')
    line_obs, = ax.plot([], [], 'rX', markersize=15, label='Obstáculo Móvel')
    
    ax.legend(loc='lower left')

    # 3. Função de Atualização da Animação
    def update(frame):
        """ Chamada a cada frame: atualiza a posição dos objetos. """
        
        rx, ry, ox, oy = path_data[frame] 
        
        # Atualiza a posição do Robô
        line_rob.set_data([rx], [ry])
        
        # Atualiza a posição do Obstáculo
        line_obs.set_data([ox], [oy])
        
        # Atualiza o título
        ax.set_title(f"Passo {frame} - Robô: ({rx}, {ry}) | Obstáculo: ({ox}, {oy})")
        
        return line_rob, line_obs

    # 4. Criação e Execução da Animação
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(path_data), 
        interval=500, # 0.5 segundo por passo
        blit=True,     
        repeat=False   
    )

    # 5. Exibir (Requer suporte gráfico no WSL) ou Salvar
    print(f"Recompensa Total na Simulação: {total_reward:.2f}")
    plt.show() 

    # --- OPÇÃO PARA SALVAR (Se plt.show() não funcionar no WSL) ---
    # Para salvar o vídeo (necessita do ffmpeg/imagemagick)
    # print("Salvando animação em robot_dynamic_path.gif...")
    # ani.save('robot_dynamic_path.gif', writer='imagemagick', fps=5)


if __name__ == '__main__':
    # 1. Cria o ambiente
    warehouse_mdp = DynamicWarehouseMDP()
    
    # 2. Treina o agente
    trained_agent = train_agent(warehouse_mdp, episodes=5000)
    
    # 3. Simula e visualiza o resultado com animação
    animate_path(warehouse_mdp, trained_agent, num_steps=50)