

class BandCombination:
    def _init_(self):
        self.listBand = []
        self.listBandwidth = []

    def get_list_band(self):
        return self.listBand
    
    def get_list_bandwidth(self):
        return self.get_list_bandwidth

    def set_band_combi(self, listBand, listBandwidth):
        self.listBand = listBand 
        self.listBandwidth = listBandwidth

    def add_bc_element(self, bandNum, bandwidth):
        self.listBand.append(bandNum)
        self.listBandwidth.append(bandwidth)    
