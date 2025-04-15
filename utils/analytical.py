import numpy as np

def sigma_y_point(P,x,y):
    return (-2*P*y**3)/(np.pi*(x**2+y**2)**2)

def sigma_y_uniform(P,a,x,y):
    return (-P)/(2*np.pi*a)*(((a-x)*y)/((a-x)**2+y**2)+((x+a)*y)/((x+a)**2+y**2) + np.arctan((a-x)/y) + np.arctan((a+x)/y))