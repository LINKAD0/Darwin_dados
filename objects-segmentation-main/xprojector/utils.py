

import numpy as np
import math


def cubemap_angles(delta_lamb,delta_phi):
    faces = {'front':(0,0),'right':(np.pi/2,0),'back':(np.pi,0),
                  'left':(-np.pi/2,0),'top':(0,np.pi/2),'bottom':(0,-np.pi/2)}
    angles=[]
    for k,v in faces.items():
        l,p = v
        l+=delta_lamb
        p+=delta_phi
        angles.append((l,p))
 
    return None,angles

def fibonacci_sphere(samples=1,scanner_shadow_angle=30):
    
    points = []
    angles=[]
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        lamb=np.arctan2(x,z)
        phi=np.arctan2(y,np.sqrt(x**2+z**2))
        
        #if (phi<0)&(phi<(np.pi/2 - scanner_shadow_angle*(np.pi/180))):
        #    continue
        
        angles.append((lamb,phi))

        points.append((x, y, z))

    return points,angles


