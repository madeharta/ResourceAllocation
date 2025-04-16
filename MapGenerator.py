import BaseStation
import MobileStation
import numpy as np

ENB_RADIUS = 500
GNB_RADIUS = 200
N_GNB = 10
LIST_POS_GNB = [[-50,-50],[-400,-100],[-100,20],[-300,300],[50,400],[200,200],[20,20],[300,-250],[200,400],[100,300]]
USER_SPEED = 1 #1 m/s

class MapGenerator: 
    def __init__(self, nENB, nGNB, nUE):
          self.nENB = nENB
          self.nGNB = nGNB
          self.nUE = nUE

          self.list_BS = []
          self.list_UE = []
    
    def generate_BS(self):
        self.list_BS.append(BaseStation(0, 0, self.nUE, [0, 0], ENB_RADIUS))
        for i in range(self.nGNB):
            self.list_BS.append(BaseStation(i, 1, self.nUE, LIST_POS_GNB[i], GNB_RADIUS))
    
    def generate_UE(self):
        for i in range(self.nUE):
            pos = self._random_point_in_circle(ENB_RADIUS)
            self.list_UE.append(MobileStation(i-1, pos, 3, -1))

    def get_list_bs(self):
        return self.list_BS

    def get_list_ue(self):
        return self.list_UE        
         
    def _random_point_in_circle(self, R):
        # Generate a random angle and a random radius (square root for uniform distribution)
        theta = np.random.uniform(0, 2 * np.pi)
        r = R * np.sqrt(np.random.uniform(0, 1))
        
        # Convert polar to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y])
