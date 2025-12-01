# DynamicState.py

class DynamicState:
    """
    Classe wrapper para o estado (rx, ry, ox, oy) que adiciona o método
    is_terminal() que o simple-rl exige para rodar experimentos.
    
    rx, ry: Posição do Robô
    ox, oy: Posição do Obstáculo
    """
    def __init__(self, state_tuple, is_goal):
        # Armazena a tupla de 4 elementos no atributo 'data'
        self.data = state_tuple
        self._is_terminal = is_goal

    def is_terminal(self):
        """ Método exigido pelo simple-rl para verificar o fim do episódio. """
        return self._is_terminal
    
    # Métodos para comparação (hash e eq) são essenciais para o Q-Learning.
    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        """ Permite a comparação correta entre objetos DynamicState e tuplas. """
        if not isinstance(other, DynamicState):
            return self.data == other
        return self.data == other.data
    
    def __str__(self):
        return f"R:({self.data[0]}, {self.data[1]}) O:({self.data[2]}, {self.data[3]})"