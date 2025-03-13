import numpy as np

NOISE_POWER = -174
SUBCHANNEL_BANDWIDTH = 208333.3333
DECORRELATED_DISTANCE = 50
LOG_NORMAL_MEAN = 0
LOG_NORMAL_STD_MS = 8
LOG_NORMAL_STD_RS = 6
USER_SPEED = 6
CARRIER_FREQ = 2e9
RAYLEIGH_SCALE = 1.0
BS_LTE_PA_POWER = 40
BS_NR_PA_POWER = 40
K_FACTOR = 10.0
MINIMUM_SINR = 0.
BS_LTE_TYPE = 0
BS_NR_TYPE = 1

def watt_to_db(self, watt):
    return (10*np.log10(watt))
    
def db_to_watt(self, db):
    return (np.pow(10, db/10))

class ChannelGainGenerator:
    def _init_(self):
          print("intisialization")
    
    def _get_awgn_noise(self):
        # 208333.333 (208kHz) / 1000 (Watt);
        return self._db_to_watt(NOISE_POWER)*SUBCHANNEL_BANDWIDTH/1000.0

    def get_channel_gain(self, bc, tx, rx):
        if tx.get_type() == BS_LTE_TYPE:
            pathLoss = -1 * (31 + 40*np.log10(tx.get_distance(rx)))
        else:
            pathLoss = -1 * (31 + 40*np.log10(tx.get_distance(rx)))
            
        timeOffset = tx.fading_to_ms[rx.get_id()]
        retVal = pathLoss + rx.getShadowing()

        mf = rx.getMultipath(bc, timeOffset)

        return  (retVal + watt_to_db(mf))
    
    def calculate_receive_power(self, txPower, tx, rx, SCID):
        if (txPower > 0) :
            return db_to_watt(self.get_channel_gain(tx, rx, SCID)) * txPower
        else :
            return 0

class MultiPathGenerator: 
    def __init__(self):
        self.w_D = any 		    #
        self.Ts	= any		    # Sampling time(tap_resolution)
        self.N0 = 16	    # Number of oscillators: 16
        self.N	= 4*self.N0 + 2
        self.prev_time = any
        self.sampling_freq = any
        self.k_factor = any
	
    def multi_path_generator(self) :
        return db_to_watt(K_FACTOR)
    """
	public void initMultiPathFading(double UserSpeed, double SamplingFreq){
		double	Max_Doppler_spread;
		int		InitShift;

		// Parameters
		Max_Doppler_spread		= UserSpeed*(CONST.STANDARD.CARRIER_FREQ)/3e8;	// [Hz]
		N0	= 16;// Using only one filter (16 oscillators)
		N	= 4*N0 + 2;
		Ts  = 1./SamplingFreq;
		sampling_freq = SamplingFreq;

		// w_D: maximum doppler shift
		// w_D = 2PAI*fdmax=2PAI*(speed/3600. *1/lamda)=2PAI*(speed/3600. *freq/c),
		w_D		= 2.* Math.PI * Max_Doppler_spread;

		// Shifting time for skipping the initial period
		InitShift	= (int) (N0*SamplingFreq/Max_Doppler_spread);
		prev_time	= Ts * InitShift;
		// Radomize the starting sample time
		prev_time	+= Ts * InitShift * Math.random() + Ts * Math.random();
	}
	"""
    def get_multi_path_fading_rician(k_factor, currTime):
		#alpha = initialPhase;
        Alpha = 0.0
		#double t1, t;
        M = 8
        N = 34.0           #N = 2.0*(2.0 * M + 1.0);
        dopplerFreq = 4.0    #//Max dopper frequency : 4 Hz

        t1 = (currTime/(2.0*np.pi*dopplerFreq))
        t = currTime - (2.0*np.pi*dopplerFreq *t1)
        t = currTime

		#Xreal = sqrt(2.0) * cos(alpha) * cos(t);
		#Ximag = sqrt(2.0) * sin(alpha) * cos(t);
        Xreal = 0.0
        Ximag = np.sqrt(2.0) * np.cos(t)
        for i in range(M): 
            Beta = i * np.pi /(M+1)
            low_frequency = 2 * np.pi * dopplerFreq *  np.cos(2.0*np.pi*i/N)
            Xreal += 2 * np.cos(Beta) * np.cos(low_frequency* t)               #WaveformForReal
            Ximag += 2 * np.sin(Beta) * np.cos(low_frequency * t)              #WaveformForImaginary

        LOS_scaling = np.sqrt(k_factor/(k_factor+1.))
        if (k_factor == 0):
            LOS_scaling = 0
		
        multi_scaling = np.sqrt(1./(k_factor+1.))
        Xreal = LOS_scaling + multi_scaling*Xreal
        Ximag = multi_scaling*Ximag

        # return RiceanChannelGain
        return (Xreal*Xreal + Ximag*Ximag)/(2*M +1)
	
    def get_multi_path_fading_jakes(self, speed, currTime):
        alpha = 0.0
        M = 8
        N = 34.0
        dopplerFreq = CARRIER_FREQ * speed / 300000000
        t1 = (currTime/(2.0*np.pi*dopplerFreq))
		#t = currTime - (2.0*np.pi*dopplerFreq*t1);
        t = currTime

		#Xc=Math.cos(alpha)*Math.cos(t)*Math.sqrt(2.);
		#Xs=Math.sin(alpha)*Math.cos(t)*Math.sqrt(2.);
		
        Xc = 0.0
        Xs = np.sqrt(2.0) * np.cos(t)
        for i in range(M):
            beta = i * np.pi /(M+1)
            low_frequency = 2 * np.pi * dopplerFreq *  np.cos(2.0*np.pi*i/(N))
            Xc += 2 * np.cos(beta) * np.cos(low_frequency * t)
            Xs += 2 * np.sin(beta) * np.cos(low_frequency * t)
		
        #return fadingChannelGain;
        return (Xc*Xc + Xs*Xs)/(2*M +1)
		#cout << "fadingChannelGain " << fadingChannelGain  << " dBm is " << dB(fadingChannelGain) << " speed is " << speed << " doppler freq. " << dopplerFreq << " t is " << t <<  " at " << currTime << " at " << currTime << endl;
	
	#calculates multi-path fading gain through single-tap Jake's model in watt
    def get_multi_path_fading(k_factor, speed, currTime):
        N0 = 8
        alpha = 0

        Wm=2.* np.pi * CARRIER_FREQ*speed*1000./3600.0/300000000.0 	# speed is in km/h so it needs to be converted to m/s
        t1=(currTime/Wm)
		#t=currTime-Wm*t1
        t=currTime
        N=(4*N0+2)
        Xc=np.cos(alpha)*np.cos(t)*np.sqrt(2.)
        Xs=np.sin(alpha)*np.cos(t)*np.sqrt(2.)
        for i in range(N0):
            beta=i*np.PI/(N0+1)
            w = Wm * np.cos(2*np.pi*i/N)
            Xc += 2 * np.cos(beta) * np.cos(w * t)
            Xs += 2 * np.sin(beta) * np.cos(w * t)
		
        Xc *= 1./np.sqrt(2*N0+1)
        Xs *= 1./np.sqrt(2*N0+1)

		#Convert to Rician fading
        LOS_scaling = np.sqrt(k_factor/(k_factor+1.))
        multi_scaling = np.sqrt(1./(k_factor+1.))
        Xc = LOS_scaling * np.cos(Wm * t) + multi_scaling*Xc
        Xs = LOS_scaling * np.sin(Wm * t) + multi_scaling*Xs

        return (Xc*Xc+Xs*Xs)
