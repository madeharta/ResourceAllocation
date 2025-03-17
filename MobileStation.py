import numpy as np

LOG_NORMAL_MEAN = 0
LOG_NORMAL_STD_MS = 8
DECORRELATED_DISTANCE = 50


class MobileStation:
    def __init__(self, id, pos, speed, mn):
        self.id = id
        self.pos = pos
        self.shadowingDB = np.random.normal(LOG_NORMAL_MEAN, LOG_NORMAL_STD_MS)
        self.prevPosUpdateShadowing = pos
        self.speed = speed
        self.mn = mn

    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos

    def set_mn(self, mn):
        self.mn = mn

    def update_pos(self, pos):
        self.pos = pos

    def update_bc(self, bcId):
        self.bcId = bcId        

    def _get_distance(self, pos1, pos2):
        return np.sqrt(np.power(pos1[0]-pos2[0],2)+np.power(pos1[1]-pos2[1],2))

    def update_shadowing(self):
        movingDistance = self._get_distance(self.prevPosUpdateShadowing, self.pos)

        largestDB		= 2.*np.abs(self.shadowingDB)
        smallestDB		= -2.*np.abs(self.shadowingDB)

        Factor_c = np.exp(-1.0 * (movingDistance/DECORRELATED_DISTANCE) * np.log(2.0))
        self.shadowingDB = Factor_c * self.shadowingDB + np.sqrt(1.-Factor_c*Factor_c) * np.random.normal(LOG_NORMAL_MEAN, LOG_NORMAL_STD_MS) #getGaussian();

        if (self.shadowingDB > largestDB):
            self.shadowingDB = largestDB
        if (self.shadowingDB < smallestDB): 
            self.shadowingDB = smallestDB
 
        self.prevPosUpdateShadowing = self.pos

    def auto_update_pos(self, deltaTime):
        movingDistance = deltaTime * self.speed * 1000/3600		# speed is in km/h so it needs to be converted to m/s
        newX = self.pos[0] + (movingDistance*np.cos(self.getMovingDirection()))
        newY = self.pos[1] - (movingDistance*np.sin(self.getMovingDirection()))

        if (self._get_distance(self.mn.get_pos(), np.array(newX, newY)) > self.bs.get_radius()):	
            # if the user move out to the cell, place it again in cell randomly
            self.update_pos(self.mn.get_random_point())
        else:
            self.update_pos(np.array(newX, newY))