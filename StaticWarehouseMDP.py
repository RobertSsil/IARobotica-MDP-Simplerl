# File: StaticWarehouseMDP.py
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State 

# Definições do Ambiente 7x6
WIDTH, HEIGHT = 7, 6 
GOAL_POS = (WIDTH - 1, 0) # (6, 0) - Alvo
START_POS = (0, HEIGHT - 1) # (0, 5) - Início

# Posições dos obstáculos fixos
OBSTACLE_POS = [
    (1, 5), (1, 3), (1, 1), 
    (3, 4), (4, 4), 
    (5, 1), (6, 1), 
]

def manhattan_distance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


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
                     gamma=0.95) 

    def is_goal_state(self, state):
        return state.data in self.goal_locs

    def is_obstacle(self, loc):
        return loc in self.obstacles

    def get_reward(self, state, action, next_state):

        if self.is_goal_state(next_state):
            return 500.0

        reward = -1.0

        curr = state.data
        nxt = next_state.data
        goal = self.goal_locs[0]

        if nxt == curr:
            reward -= 3.0

        if self.is_obstacle(nxt):
            reward -= 10.0

        d_before = manhattan_distance(curr, goal)
        d_after = manhattan_distance(nxt, goal)

        if d_after < d_before:
            reward += 2.0
        else:
            reward -= 2.0

        return reward
        
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