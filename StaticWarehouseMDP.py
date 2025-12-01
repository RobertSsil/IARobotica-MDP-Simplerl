from simple_rl.mdp.MDPClass import MDP
import random
import numpy as np
from simple_rl.mdp.StateClass import State 

# Definições do Ambiente 7x6 (baseado na imagem)
WIDTH, HEIGHT = 7, 6 
GOAL_POS = (WIDTH - 1, 0) # (6, 0) - Canto inferior direito (Alvo no MDP)
START_POS = (0, HEIGHT - 1) # (0, 5) - Canto superior esquerdo (Robô no MDP)

# Posições dos obstáculos fixos (Baseado na imagem anexa, com (0,0) no canto inferior esquerdo)
OBSTACLE_POS = [
    (1, 5), # Coluna 1, linha superior
    (1, 3), # Coluna 1, linha intermediária
    (1, 1), # Coluna 1, linha inferior

    (3, 4), # Coluna 3
    (4, 4), # Coluna 4
    
    (5, 1), # Coluna 5 (inferior)
    
    (6, 1), # Coluna 6 (próximo ao alvo)
] 


class StaticWarehouseMDP(MDP):
    def __init__(self, width=WIDTH, height=HEIGHT, goal_locs=[GOAL_POS], init_loc=START_POS):
        
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        self.obstacles = OBSTACLE_POS
        self.init_loc = init_loc

        actions = ["NORTH", "SOUTH", "EAST", "WEST"]
        
        init_state = State(init_loc)
        
        MDP.__init__(self, 
                     actions=actions, 
                     init_state=init_state,
                     transition_func=self.get_next_state,
                     reward_func=self.get_reward,
                     gamma=0.9) 

    def is_goal_state(self, state):
        return state.data in self.goal_locs

    def is_obstacle(self, loc):
        return loc in self.obstacles

    def get_reward(self, state, action, next_state):
        
        if self.is_goal_state(next_state): 
            return 10.0
            
        return -0.1
        
    def get_next_state(self, state, action):
        
        rx, ry = state.data 
        
        if self.is_goal_state(state):
            return state

        r_next_x, r_next_y = rx, ry
        
        if action == "NORTH": r_next_y = min(self.height - 1, ry + 1)
        elif action == "SOUTH": r_next_y = max(0, ry - 1)
        elif action == "EAST": r_next_x = min(self.width - 1, rx + 1)
        elif action == "WEST": r_next_x = max(0, rx - 1)
        
        new_loc = (r_next_x, r_next_y)
        
        if self.is_obstacle(new_loc):
            final_loc = (rx, ry)
        else:
            final_loc = new_loc

        return State(final_loc)

    def get_init_state(self):
        return State(self.init_loc)

    def get_actions(self):
        return ["NORTH", "SOUTH", "EAST", "WEST"]
    
    def get_parameters(self):
        return {"width": self.width, "height": self.height, "obstacles": self.obstacles}