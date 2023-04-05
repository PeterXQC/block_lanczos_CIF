import numpy as np

# define the circular part of the D contour
# added unused variable angle for compatibility
def Γ(t, angle, r, λmin, c):
    
    z = r*np.exp(1j*t)+c
    dz = r*1j*np.exp(1j*t)
    
    return z,dz

# define the straight part of the D contour
# 0 <= t <= 1 is the bottom half straight line
# 1 <= t <= 2 is the top half straight line
def Γl(t, angle, r, λmin, c):
    
    z = c+t*r*np.exp(1j*(angle))
    dz = r*np.exp(1j*(angle))
    
    return z,dz