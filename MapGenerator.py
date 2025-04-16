import BaseStation
import numpy as np

ENB_RADIUS = 500
GNB_RADIUS = 200
N_GNB = 10
LIST_POS_GNB = [[-50,-50],[-400,-100],[-100,20],[-300,300],[50,400],[200,200],[20,20],[300,-250],[200,400],[100,300]]
USER_SPEED = 1 #1 m/s

class MobileStation: 
    def __init__(self, nENB, nGNB, nUE):
          self.nENB = nENB
          self.nGNB = nGNB
          self.nUE = nUE

    def generate_nENB(self):
        pos = [0, 0]
        return BaseStation(0, 0, self.nUE, pos, ENB_RADIUS)
    
    def generate_nGNB(self):
        list_gNB = []
        for i in range(self.nGNB):
            list_gNB.append(BaseStation(i, 1, self.nUE, LIST_POS_GNB[i], GNB_RADIUS))

        return list_gNB
    
    def generate_nUE(self):
        list_nUE = []
        for i in range(self.nUE):
            pos = self._random_point_in_circle(ENB_RADIUS)
            list_nUE.append(MobileStation(i-1, pos, 3, -1))

         
    def _random_point_in_circle(self, R):
        # Generate a random angle and a random radius (square root for uniform distribution)
        theta = np.random.uniform(0, 2 * np.pi)
        r = R * np.sqrt(np.random.uniform(0, 1))
        
        # Convert polar to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y])
