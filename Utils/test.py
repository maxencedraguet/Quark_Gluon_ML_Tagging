import os
import sys
import json
from typing import Dict

import numpy as np
import pandas as pd
import math

from Tools import tree_plotter

def get_angle_between_momenta(mom1_vec, mom2_vec):
    """
        Returns the angle between two momentas of the shape:
        mom1_vec = [mom1.px, mom1.py, mom1.pz]
        """
    # Normalise
    mom1_vec = mom1_vec / np.linalg.norm(mom1_vec)
    mom2_vec = mom2_vec / np.linalg.norm(mom2_vec)
    cos_theta = np.longdouble(np.dot(mom1_vec, mom2_vec))
    if(abs(cos_theta- 1.0)<1e-12):
        return 0
    elif(abs(cos_theta + 1)<1e-12):
        return math.pi
    else:
        return np.arccos(cos_theta)

def get_plane_of_momenta(mom1_vec, mom2_vec):
    """
        Returns the angle of the plane where the two momenta lie.
        This angle is defined with respect to a fixed direction (along the py component)
        and the direction of mom2 is taken to be purely along z (will be fed the mother).
        
        Shape of momenta: mom1 = [mom1.px, mom1.py, mom1.pz]
        """
    if(abs(get_angle_between_momenta(mom1_vec, mom2_vec))< 1e-12):
        return 0.0  # protection in case the particles are extremely close
    
    # First fixed direction in cartesian plane
    y_direction = [0.0, 1.0, 0.0]
    # Set the mother, played by mom2, as the reference (mothers will be purely along e_z
    e_z_direction = mom2_vec / np.linalg.norm(mom2_vec)
    
    # From mother component, get dextrorotatory system
    e_x_direction = np.cross(y_direction, e_z_direction)
    e_x_direction = e_x_direction / np.linalg.norm(e_x_direction)
    
    e_y_direction = np.cross(e_z_direction, e_x_direction)
    e_y_direction = e_y_direction / np.linalg.norm(e_y_direction)
    
    # We now have the coordinate system, project mom1 in e_x, e_y system with components in the cartesian one
    projection_mom1_xy = [0.0] * 3
    projection_mom1_xy[0] = mom1_vec[0] - e_z_direction[0] * np.dot(e_z_direction, mom1_vec)
    projection_mom1_xy[1] = mom1_vec[1] - e_z_direction[1] * np.dot(e_z_direction, mom1_vec)
    projection_mom1_xy[2] = mom1_vec[2] - e_z_direction[2] * np.dot(e_z_direction, mom1_vec)
    # Get the angle between this projection and e_y
    phi_y = get_angle_between_momenta(projection_mom1_xy, e_y_direction)
    phi_x = get_angle_between_momenta(projection_mom1_xy, e_x_direction)
    
    if(phi_y > math.pi/2):
        phi_x = 2 * math.pi - phi_x
    return phi_x, phi_y



json_file = "../example_JUNIPR_data_CA.json"

with open(json_file) as json_file:
    data_array = json.load(json_file)['JuniprJets'] #a list of dictionnaries
data_item = data_array[2] # take the first
path = "example_JUNIPR_data_CA_tree/tree_"
for count, item in enumerate(data_array):
    tree_plotter(item, path + str(count))
