

class BandCombination:
    def _init_(self):
        self.listBand = []
        self.listBandwidth = []
        self.listFreq = []
        self.listBS = []

    def get_list_band(self):
        return self.listBand
    
    def get_list_bandwidth(self):
        return self.listBandwidth

    def get_list_freq(self):
        return self.listFreq

    def get_list_bs(self):
        return self.listBS

    def set_band_combi(self, listBand, listBandwidth, listFreq, listBS):
        self.listBand = listBand 
        self.listBandwidth = listBandwidth
        self.listFreq = listFreq
        self.listBS = listBS

    def add_bc_element(self, bandNum, bandwidth, freq, bs):
        self.listBand.append(bandNum)
        self.listBandwidth.append(bandwidth) 
        self.listFreq.append(freq)   
        self.listBS.append(bs)
