#from invisible_cities.io.mcinfo_io import load_mchits
import numpy as np
cimport numpy as np
from scipy import optimize


# pos is in mm
# generare the number of electrons from the E/22eV to mimic the ionization
# then ge tthe number of electrons by taking a gaussin with mean the # and sigma root#
# distrbute the electrons in a 100 micron sphere (uniformly? sounds right)

cdef double Ionization = 22/1e6

def MC_Z(MC_HITs):
    cdef int qw, ll
    cdef double zz
    ll = len(MC_HITs)
    zz = 0
    for qw in range(0,ll):
        zz+=MC_HITs[qw].pos[2]
    zz = zz/ll
    return zz

def MC_to_electrons(MC_HIT,double Space_sigma):
    cdef double si, X_mu, Y_mu, Z_mu, znew, ynew, xnew
    cdef int mu ,po
    #Space_sigma = 0.1 # 0.1 should be 100 microns
    mu = int(MC_HIT.E/Ionization) # piicks the number of electrons
    si = np.sqrt(mu) # decides there error
    New_mu = int(np.random.normal(mu, si, 1))# gererates a gaussian for that with the number of electrons
    X_mu =MC_HIT.pos[0] ; Y_mu = MC_HIT.pos[1] ; Z_mu = MC_HIT.pos[2]
    cdef list New_pos = []
    for po in range(New_mu):
        xnew = np.random.uniform(X_mu-Space_sigma, X_mu+Space_sigma, 1)[0]
        ynew = np.random.uniform(Y_mu-Space_sigma, Y_mu+Space_sigma, 1)[0]
        znew = np.random.uniform(Z_mu-Space_sigma, Z_mu+Space_sigma, 1)[0]
        New_pos.append([xnew,ynew,znew])
    return New_pos

def Event_to_electrons(MC_HITs,double Space_sigma):
    cdef int qw
    cdef list Event_electrons_list = []
    for qw in range(0,len(MC_HITs)):
        hold = MC_to_electrons(MC_HITs[qw],Space_sigma)
        Event_electrons_list += hold
    cdef np.ndarray[np.float64_t ,ndim=2] Event_electrons = np.array(Event_electrons_list)
    return Event_electrons


def Diffuser(Event_electrons,Gas_props,int Number_electrons):
    cdef int ty
    cdef double Dif_Tran_star  = Gas_props[0]  #*1e-2
    cdef double Dif_Long_star  = Gas_props[1]  #*1e-2
    cdef double Drift_Vel      = Gas_props[2]  #units mm/mus
    cdef double Pressure       = Gas_props[3]  #units of bar
    cdef double Life_Time      = Gas_props[4]  #units mus

    cdef list Diffused_list = []
    
    for ty in range(0,len(Event_electrons)):
        X = Event_electrons[ty][0]
        Y = Event_electrons[ty][1]
        Z = Event_electrons[ty][2]

        Drift_distance = abs(1000 + Z)
        DT = Dif_Tran_star*np.sqrt(Drift_distance*1e-1/Pressure)*1e-3
        DL = Dif_Long_star*np.sqrt(Drift_distance*1e-1/Pressure)*1e-3
        #Drift_time     = Drift_distance/(Drift_Vel)
        #Life_time_num = np.random.exponential(Drift_time/Life_Time,1)[0]
        #if Life_time_num <= Drift_time/Life_Time:

        xnew = np.random.normal(X, DT, 1)[0]
        ynew = np.random.normal(Y, DT, 1)[0]
        znew = np.random.normal(Z, DL, 1)[0]
        Diffused_list.append([xnew,ynew,znew])
    # this should pick out only the number of electrons that corrospond with the smered energy
    lll = np.random.choice(len(Diffused_list),(Number_electrons-2), replace=False)
    cdef np.ndarray[np.float64_t ,ndim=2] Diffused = np.array(Diffused_list)[lll]
    return Diffused


