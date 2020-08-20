 #############################################################################
#
# JUNIPR_distributions_plot.py
#
# Some usefull info to plot the JUNIPR probability distributions:
#
# Author -- Maxence Draguet (11/08/2020)
#
#############################################################################
import numpy as np

distributions_to_plot = ['ending', 'mother_id', 'branch_z', 'branch_theta', 'branch_phi', 'branch_delta']


plot_xlabels = {
               'ending'           : 'State length',
               'mother_id'        : 'Mother index in energy sorted list',
               'branch_z'         : 'Branching $z$',
               'branch_theta'     : 'Branching $\\theta$',
               'branch_phi'       : 'Branching $\\phi$',
               'branch_delta'     : 'Branching $\\delta$'
                }

plot_ylabels = {
                'ending'          : 'probability to end',
                'mother_id'       : 'probability to branch',
                'branch_z'        : 'probability',
                'branch_theta'    : 'probability',
                'branch_phi'      : 'probability',
                'branch_delta'    : 'probability'
                }

plot_axis = {
                'ending'           : [0, 60, 0, 1],
                'mother_id'        : [-0.5, 15.5, 0, 1],
                'branch_z'         : None,
                'branch_theta'     : None,
                'branch_phi'       : [0, 2* np.pi, 0, 0.15],
                'branch_delta'     : None
                }

plot_xscale = {
                'ending'           : None,
                'mother_id'        : None,
                'branch_z'         : 'log',
                'branch_theta'     : 'log',
                'branch_phi'       : None,
                'branch_delta'     : 'log'
                }

plot_end_setting = {
                'ending'           : {'alpha':0.5, 'width':1},
                'mother_id'        : {'alpha':0.5, 'width':1},
                'branch_z'         : {'alpha':0.5, 'align':'edge'},
                'branch_theta'     : {'alpha':0.5, 'align':'edge'},
                'branch_phi'       : {'alpha':0.5, 'align':'edge'},
                'branch_delta'     : {'alpha':0.5, 'align':'edge'}
            }

"""
plot_xbins  = {
               'ending'           : 56,
               'mother_id'        : 60,
               'branch_z'         : 1000,
               'branch_theta'     : 50,
               'branch_phi'       : 120,
               'branch_delta'     : 25
               }
"""
plot_xticks  = {
               'ending'           : [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
               'mother_id'        : None,
               'branch_z'         : ([0.002, 0.01, 0.1, 0.5], [0.002, 0.01, 0.1, 0.5]),
               'branch_theta'     : ([0.05, 0.1, 0.3, 1, np.pi/2, np.pi], [0.05, 0.1, 0.3, 1, '$\pi/2$', '$\pi$']),
               'branch_phi'       : ([0, np.pi, 2* np.pi], [0,'$\pi$', '$2\pi$']),
               'branch_delta'     : ([2e-5, 2e-4, 0.01, 0.1, 0.8], [2e-5, 2e-4, 0.01, 0.1, 0.8])
               }

def prepare_date(data, label):
    if label == 'ending' or label == 'mother_id':
        axis = range(len(data))
        return axis, data
    
    elif label == 'branch_phi':
        x_axis = np.linspace(0,1,11) * 2 * np.pi
        width  = np.diff(x_axis)
        return x_axis[:-1], data, width

    elif label == 'branch_z':
        parameter = 2
        limit = 1/2

    elif label =='branch_theta':
        parameter = 50
        limit =  np.pi

    elif label == 'branch_delta':
        parameter = 5000
        limit =  np.pi

    x_axis = (np.exp( np.log( 1 + parameter * limit) * np.linspace(0,1,11)) - 1) / parameter
    width  = np.diff(x_axis)
    return x_axis[:-1], data, width
