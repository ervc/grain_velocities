import numpy as np

import constants as const

# some constants and analytic functions
R0 = 5.2*const.AU
M_sun = const.MSUN
G = const.G
rho_s = 2.0
TIME = 1/np.sqrt(G*M_sun/R0/R0/R0)
R_jup = 69911*1e5

def get_scaleheight(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    return 0.05*r*(r/R0)**(1/4)

def get_Omega_K(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    return np.sqrt(G*M_sun/r)/r

def get_vkep(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    return np.sqrt(G*M_sun/r)

def get_soundspeed(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    H = get_scaleheight(x,y,z)
    Omega_K = get_Omega_K(x,y,z)
    return H*Omega_K

def get_disk_temperature(x,y,z):
    ### Isothermal
    r = np.sqrt(x*x + y*y + z*z)
    T0 = 188
    return T0*(r/R0)**(-1/2)