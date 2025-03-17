import numpy as np

RAND_MAX = 32787


class BaseStation:
    def __init__(self, id, type, noUE, pos, radius):
        self.id = id
        self.type = type
        self.noUE = noUE
        self.pos = pos
        self.fading_bs_to_ms = np.random(noUE,)*RAND_MAX
        self.radius = radius

    def get_pos(self):
        return self.pos
    
    def get_id(self):
        return self.id
    
    def get_type(self):
        return self.id
    
    def get_fading_info(self, ueID):
        return self.fading_bs_to_ms[ueID]
    
    def get_radius(self):
        return self.radius
    
    def _rand_AB(self, a, b):
        return (a+(b-a)*np.random.random_integers())
        
    def get_random_point(self):	
        tempX = self._rand_AB(-1.0 * self.radius, 0.5 * self.radius)
        tempY = self._rand_AB(-np.sqrt(3.)/2.*self.radius, np.sqrt(3.)/2.*self.radius)
        if (tempX < -0.5 * self.radius):
            if (tempY > (np.sqrt(3.)*(tempX+self.radius))):
                tempX += 1.5 * self.radius
                tempY -= np.sqrt(3.)/2.*self.radius
            
            if (tempY < (-np.sqrt(3.)*(tempX+self.radius))):
                tempX += 1.5 * self.radius
                tempY += np.sqrt(3.)/2.*self.radius
            

        return (np.array(self.pos[0]+tempX, self.pos[1]+tempY))