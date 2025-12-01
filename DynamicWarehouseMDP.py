from simple_rl.mdp.MDPClass import MDP
import random
import numpy as np
from DynamicState import DynamicState # Importa a classe de estado corrigida

# --- 1. CONFIGURAÇÕES DO AMBIENTE ---
WIDTH, HEIGHT = 5, 5
GOAL_POS = (WIDTH - 1, HEIGHT - 1)
START_POS = (0, 0)
OBSTACLE_START_POS = (2, 2)

class DynamicWarehouseMDP(MDP):
    """
    Processo de Decisão de Markov (MDP) Dinâmico para Robô de Armazém.
    O estado S é um objeto DynamicState que encapsula a tupla (rx, ry, ox, oy).
    """
    def __init__(self, width=WIDTH, height=HEIGHT, goal_locs=[GOAL_POS], init_loc=START_POS):
        
        # --- CORREÇÃO APLICADA AQUI: Definir atributos essenciais primeiro ---
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        # ---------------------------------------------------------------------

        actions = ["NORTH", "SOUTH", "EAST", "WEST"]
        
        all_state_list = self._get_all_states(width, height)
        init_state = self._get_state_from_loc(init_loc, OBSTACLE_START_POS)
        
        # Inicialização do MDP com funções obrigatórias
        MDP.__init__(self, 
                     actions=actions, 
                     init_state=init_state,
                     transition_func=self.get_next_state,
                     reward_func=self.get_reward,
                     gamma=0.9) 
        
        self.all_states = all_state_list # Esta variável não causa problemas de ordem

    def _get_all_states(self, w, h):
        """ Gera todos os estados possíveis (Robô x Obstáculo) como DynamicState """
        states = []
        for rx in range(w):
            for ry in range(h):
                for ox in range(w):
                    for oy in range(h):
                        state_tuple = (rx, ry, ox, oy)
                        # self.goal_locs agora existe quando esta linha é chamada
                        is_goal = self.is_goal_state_tuple(state_tuple) 
                        states.append(DynamicState(state_tuple, is_goal))
        return states
    
    def _get_state_from_loc(self, rob_loc, obs_loc):
        """ Cria o DynamicState a partir das coordenadas """
        rx, ry = rob_loc
        ox, oy = obs_loc
        state_tuple = (rx, ry, ox, oy)
        is_goal = self.is_goal_state_tuple(state_tuple)
        return DynamicState(state_tuple, is_goal)

    def is_goal_state(self, state):
        """ Verifica se o robô está no objetivo (recebe DynamicState) """
        return self.is_goal_state_tuple(state.data)

    def is_goal_state_tuple(self, state_tuple):
        """ Lógica real de terminalidade (recebe a tupla) """
        rob_loc = (state_tuple[0], state_tuple[1])
        return rob_loc in self.goal_locs # self.goal_locs agora está definido!

    # --- FUNÇÃO DE RECOMPENSA (R) ---
    def get_reward(self, state, action, next_state):
        rob_loc = (next_state.data[0], next_state.data[1])
        obs_loc = (next_state.data[2], next_state.data[3])
        
        if self.is_goal_state(next_state): 
            return 10.0
            
        if rob_loc == obs_loc:
            return -10.0
            
        if abs(rob_loc[0] - obs_loc[0]) + abs(rob_loc[1] - obs_loc[1]) <= 1:
            return -2.0
            
        return -0.1
        
    # --- FUNÇÃO DE TRANSIÇÃO (T) ---
    def get_next_state(self, state, action):
        
        rx, ry, ox, oy = state.data 
        
        if self.is_goal_state(state):
            return state

        r_next_x, r_next_y = rx, ry
        
        # Movimento do ROBÔ (Determinístico)
        if action == "NORTH": r_next_y = min(self.height - 1, ry + 1)
        elif action == "SOUTH": r_next_y = max(0, ry - 1)
        elif action == "EAST": r_next_x = min(self.width - 1, rx + 1)
        elif action == "WEST": r_next_x = max(0, rx - 1)
        
        # Movimento do OBSTÁCULO MÓVEL (Aleatório)
        if random.random() < 0.5:
            obs_moves = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
            dx, dy = obs_moves[random.choice(list(obs_moves.keys()))]
        else:
            dx, dy = 0, 0
            
        o_next_x = max(0, min(self.width - 1, ox + dx))
        o_next_y = max(0, min(self.height - 1, oy + dy))

        # Retorna o novo DynamicState
        new_state_tuple = (r_next_x, r_next_y, o_next_x, o_next_y)
        is_goal = self.is_goal_state_tuple(new_state_tuple)
        return DynamicState(new_state_tuple, is_goal)

    # --- Funções do simpleRL ---
    def get_init_state(self):
        return self._get_state_from_loc(START_POS, OBSTACLE_START_POS)

    def get_actions(self):
        return ["NORTH", "SOUTH", "EAST", "WEST"]
    
    def get_parameters(self):
        return {"width": self.width, "height": self.height}