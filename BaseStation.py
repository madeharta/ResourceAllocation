import numpy as np

RAND_MAX = 32787

class BaseStation:
    def __init__(self, id, type, noUE, position):
        self.id = id
        self.type = type
        self.noUE = noUE
        self.position = position
        self.fading_bs_to_ms = np.random(noUE,)*RAND_MAX

    def get_position(self):
        return self.position
    
    def get_id(self):
        return self.id
    
    def get_type(self):
        return self.id
    
    