import numpy as np
import math
import BandCombination

LOG_NORMAL_MEAN = 0
LOG_NORMAL_STD_MS = 8
SHANNON_MODUL_GAP = 1
DECORRELATED_DISTANCE = 50


class MobileStation:
    def __init__(self, id, pos, speed, movDir, mn, bs, bc):
        self.id = id
        self.pos = pos
        self.shadowingDB = np.random.normal(LOG_NORMAL_MEAN, LOG_NORMAL_STD_MS)
        self.prevPosUpdateShadowing = pos
        self.speed = speed
        self.movDir = np.radians(movDir)
        self.mn = mn
        self.bs = bs
        self.bc = bc

    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos

    def get_bs(self):
        return self.bs

    def get_bc(self):
        return self.bc

    def set_mn(self, mn):
        self.mn = mn

    def set_bs(self, bs):
        self.bs = bs

    def set_bc(self, bc):
        self.bc = bc

    def update_pos(self, pos):
        self.pos = pos

    def update_bc(self, bcId):
        self.bcId = bcId        

    def get_moving_direction(self):
        return self.movDir

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
        newX = self.pos[0] + (movingDistance*np.cos(self.movDir))
        newY = self.pos[1] - (movingDistance*np.sin(self.movDir))

        if (self._get_distance(self.mn.get_pos(), np.array(newX, newY)) > self.bs.get_radius()):	
            # if the user move out to the cell, place it again in cell randomly
            self.update_pos(self.mn.get_random_point())
        else:
            self.update_pos(np.array(newX, newY))

    def get_throughput(self):
        listBand = self.bc.get_list_band()
        listBandwidth = self.bc.get_list_bandwidth()
        rate = 0
        # shall calculate SINR here
        sinr = 0
        for i in range(len(listBand)):
            rate += listBandwidth[i] * (math.log10(1 + SHANNON_MODUL_GAP * sinr) / math.log10(2));  

        return rate