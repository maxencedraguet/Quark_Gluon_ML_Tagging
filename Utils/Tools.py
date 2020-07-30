#############################################################################
#
# Tools.py
#
# A set of helpful method to analyse and produce results.
#
# Author -- Maxence Draguet (09/06/2020)
#
#############################################################################
import os
import itertools
import numpy as np
from sklearn import metrics
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, patches, cm

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix given in cm.
    Aims to mimic that produced by sklearn plot_confusion_matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    fmt = '.2f' if normalize else 'd'
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    return fig

def get_dictionary_cross_section():
    """
    Read and convert to dictionary of cross section
    """
    data = np.loadtxt("Backgrounds.txt", usecols=(0, 2, 3))
    cross_section_dictionary = dict()
    for row in data:
        cross_section_dictionary[str(int(row[0]))] = (row[1], row[2])
    
    return cross_section_dictionary

def write_ROC_info(filename, test, pred):
    """
    Writes in text file filename the test label, prediction proba to be used in ROC curves
    """
    with open(filename, 'w') as f:
        for test_item, pred_item in zip(test, pred):
            f.write("{}, {}\n".format(test_item, pred_item))

def ROC_curve_plotter_from_files(list_of_files, save_path):
    """
    Given a list of files.txt in list_of_files, loads them and plots the signal ROC curve.
    Entries of list_of_files should be tuples of the forme (file_name, signal_name)
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for file_name, signal_name in list_of_files:
        truth_pred, proba_pred = np.loadtxt(file_name, delimiter=',', unpack=True)
        false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(truth_pred, proba_pred)
        AUC_test = metrics.auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label='{} (area = {:.4f})'.format(signal_name, AUC_test))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(os.path.join(save_path, 'ROC_curve.png'), dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def ROC_curve_plotter_from_values(list_of_signal, save_path):
    """
    Given a list of signals in list_of_signal, plots the ROC curve.
    Entries of list_of_files should be tuples of the forme (signal_name, truth_pred, proba_pred), the last two being numpy arrays
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for signal_name, truth_pred, proba_pred in list_of_signal:
        #print(signal_name, truth_pred, proba_pred)
        false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(truth_pred, proba_pred)
        AUC_test = metrics.auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label='{} (area = {:.4f})'.format(signal_name, AUC_test))
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves')
    #plt.legend(loc='best')
    #fig.legend(loc=7)
    #fig.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.subplots_adjust(right=0.7)
    plt.savefig(os.path.join(save_path, 'ROC_curve.png'), dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

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

def circle(fig, ax, xy, radius, kwargs=None):
    """
    TAKEN FROM: https://werthmuller.org/blog/2014/circle/
    Create circle on figure with axes of different sizes.
        
    Plots a circle on the current axes using `plt.Circle`, taking into account
    the figure size and the axes units.
    
    It is done by plotting in the figure coordinate system, taking the aspect
    ratio into account. In this way, the data dimensions do not matter.
    However, if you adjust `xlim` or `ylim` after plotting `circle`, it will
    screw them up; set `plt.axis` before calling `circle`.
    
    Parameters
    ----------
    fig: figure on which the axis is set
    ax: axis on which to draw
    xy, radius, kwars :
    As required for `plt.Circle`.
        
    """
    
    # Calculate figure dimension ratio width/height
    pr = fig.get_figwidth()/fig.get_figheight()
    
    # Get the transScale (important if one of the axis is in log-scale)
    tscale = ax.transScale + (ax.transLimits + ax.transAxes)
    ctscale = tscale.transform_point(xy)
    cfig = fig.transFigure.inverted().transform(ctscale)
    
    # Create circle
    if kwargs == None:
        circ = patches.Ellipse(cfig, radius, radius*pr,
                               transform=fig.transFigure)
    else:
        circ = patches.Ellipse(cfig, radius, radius*pr,
                               transform=fig.transFigure, **kwargs)

    # Draw circle
    ax.add_artist(circ)

def draw_tree(node_dic, edge_dic, add_label_proba, title, window_extrema, path, max_value_colour = 500.0, return_plt_bool = False):
    """
    Draw the tree stored in node-edge dictionnaries with matplotlib.
    Uses a colour code for edges depending on the energy of the constituents
    Adds a text at node listing their probability as well as the global probability and the jet label.
    """
    
    #fig, axes = plt.subplots(nrows = 2)
    fig = plt.figure()
    spec = matplotlib.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[9, 1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    
    #fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    ax0.set_title(title)
    colour_map = matplotlib.cm.get_cmap('jet')
    #colour_norm = matplotlib.colors.Normalize(vmin=0.0, vmax= max_value_colour, clip= True)
    if max_value_colour < 100:
        max_value_colour = 100
    colour_norm = matplotlib.colors.LogNorm(vmin=1, vmax= max_value_colour, clip= True)
    middle_colour_range_energy = colour_norm.inverse(0.5)
    middle_colour_range_energy = int((middle_colour_range_energy +2)/10) *10    # simplify it to the lowest tenth

    #plt.gca().set_aspect('equal', adjustable='box')
    ax0.axis('auto')
    ax0.set_ylim(window_extrema[2] - 0.2, window_extrema[1] + 0.2)
    ax0.set_xlim(-1, window_extrema[0] +2)
    #ax0.get_xaxis().set_visible(False)
    #ax0.get_yaxis().set_visible(False)
    ax0.axis('off')
    gradient = np.linspace(1, 0, 256)
    gradient = np.vstack((gradient, gradient))
    
    if (abs(window_extrema[2]) > abs(window_extrema[1])):
        # jet is mostly going down, write down too
        y_location_extre_text = 0.2
    else:
        y_location_extre_text = 0.8
    
    ax0.text(0.05, y_location_extre_text, "Probability: {:.2f}".format(add_label_proba[1]), fontsize=10,  horizontalalignment='left', verticalalignment='center', zorder= 102, transform = ax0.transAxes)
    
    ax0.text(0.05, y_location_extre_text- 0.1, "Jet label: {}".format(add_label_proba[0]), fontsize=10,  horizontalalignment='left', verticalalignment='center', zorder= 102, transform = ax0.transAxes)
    

    for item in edge_dic:
        nodes, energy = edge_dic[item]
        node1, node2 = nodes
        # For colour purpose, multiply energy by 2 (finale entries have fraction of geV easily and times 2 should bring about 1 GeV. Cut everything above
        colour_edge = 1.0 - colour_norm(energy)
        # Map this colour_edge into an RGBA:
        colour_edge = colour_map(colour_edge)

        # Draw the edge
        coordinate_node1 = node_dic[node1][0]
        coordinate_node2 = node_dic[node2][0]
        x_coord = [coordinate_node1[0], coordinate_node2[0]]
        y_coord = [coordinate_node1[1], coordinate_node2[1]]
        ax0.plot(x_coord, y_coord, color=colour_edge)
    
    for item in node_dic:
        position, _, proba = node_dic[item]
        if isinstance(proba, str):
            continue
        circle(fig, ax0, (position[0], position[1]), .03, {'color':'#377eb8', 'clip_on': False, 'zorder': 100})    #, 'zorder': 100
        ax0.text(position[0], position[1], int(proba*100)/100, verticalalignment='center', horizontalalignment='center', zorder= 101, fontsize=5)
    
    ax1.imshow(gradient, aspect='auto', cmap=colour_map)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    pos = list(ax1.get_position().bounds)
    y_text = pos[1] - pos[3]/3.
    fig.text(pos[0] + pos[0]/2, y_text, "1 GeV" , va='center', ha='right', fontsize=10)
    fig.text(pos[2] + pos[0]/1.5, y_text, str(max_value_colour) + " GeV", va='center', ha='center', fontsize=10)
    fig.text( (pos[2] + pos[0]/1.5 + pos[0] + pos[0]/2) /2, y_text, str() + str(middle_colour_range_energy) + " GeV", va='center', ha='center', fontsize=10)
    
    if return_plt_bool:
        return fig

    fig.savefig(path+'.png', dpi=300, format='png')
    plt.close()
    return 0

def tree_plotter(dictionnary_info, path, segment_size = 2.0, return_plt_bool = False):
    """
    A tree plotting function. Uses branching information to plot the tree (angle between edges is the 2D angle).
    Colour is the energy of the particle associated to the branch.
    Values at node is the likelihood of observing this branching as given by the analysed model.
    
    segment_size: how long an edge is to be.
    """
    n_branchings   = dictionnary_info["n_branchings"].item()
    label          = dictionnary_info["label"].item()
    branching_list = dictionnary_info["branching"]
    particle_info  = dictionnary_info["CSJets"]
    proba_list     = dictionnary_info["probability_list"][0][1] #np.linspace(1, 0, n_branchings)
    total_proba    = dictionnary_info["probability_list"][0][0].item()
    id_mother      = dictionnary_info["CS_ID_mothers"]
    id_daughter    = dictionnary_info["CS_ID_daugthers"]
    daughter_mom   = dictionnary_info["daughter_momenta"]

    branching_list = branching_list[:n_branchings, :]
    particle_info  = particle_info[:(n_branchings*2+1), :].numpy()
    id_mother      = id_mother[:n_branchings].numpy()
    id_daughter    = id_daughter[:n_branchings, :].numpy()
    daughter_mom   = daughter_mom[:n_branchings, :].numpy()
    
    dict_edge_property = dict() # for each edge_id: a tuple connecting two nodes ID + energy.
    dict_node_property = dict() # for each node_id remembers its property (tuple for (x-y position (also a tuple), list of edge id, probability).
    dict_angles_of_part = dict()
    dict_memorise_mother_direction = dict()
    
    #node_counter = 0
    edge_counter = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for branching_count, mother_id in enumerate(id_mother):
        #print("Step {}, mother id {}".format(branching_count,mother_id))
        if branching_count == 0:
            # Retrieve the first mother's energy for the energy window and round it up to the closest 100.
            energy_max = round(particle_info[mother_id][0], -2)
            # first mother must be added to the layout (next mother will be added when they are daughter.
            dict_node_property[mother_id+1] = ((0,0), [0], "EX")          # out of list node (proba 0 means 1 since it is in log)
            dict_node_property[mother_id] = ((segment_size,0), [], proba_list[0]) # the first real node (the first mother decaying)
            dict_edge_property[0] = ((mother_id+1, mother_id), particle_info[mother_id][0])
            edge_counter += 1
            dict_angles_of_part[mother_id] = 0
            dict_memorise_mother_direction[mother_id] = 0
        
        daughter1_id, daughter2_id = id_daughter[branching_count][0], id_daughter[branching_count][1]
        branching_info             = branching_list[branching_count]
        daughter1_mom, daughter2_mom = daughter_mom[branching_count][:4], daughter_mom[branching_count][4:]
        true_daughter1_id, true_daughter2_id = daughter1_id, daughter2_id
        
        if daughter2_mom[0] > daughter1_mom[0] :
            # Weird case where second daughter is the most energetic one. Invert these wrong cases
            true_daughter1_id, true_daughter2_id = daughter2_id, daughter1_id
            # daughter 1 is now the most energetic in any case
            
        daughter1_mom, daughter2_mom = particle_info[true_daughter1_id], particle_info[true_daughter2_id]
        angle_daughter1, angle_daughter2 = branching_info[3], branching_info[1]
        
        # Get the angles
        #angle_daughter1, angle_daughter2 = branching_info[3], branching_info[1]
        mother_x, mother_y = dict_node_property[mother_id][0] # return the tuple of position
        angle_mother = dict_angles_of_part[mother_id]
        
        sign = 1
        if daughter1_mom[2] > math.pi:
            sign = -1
                #print("Mother angle {}".format(angle_mother))
                #print("Sign {} Angle 1: {} and angle 2: {}".format(sign, angle_daughter1, angle_daughter2))
        daughter_1_x = mother_x + segment_size# * np.cos(angle_daughter1)
        daughter_1_y = mother_y + segment_size * np.sin(angle_mother + sign * angle_daughter1)
        daughter_2_x = mother_x + segment_size #* np.cos(angle_daughter2)
        daughter_2_y = mother_y + segment_size * np.sin(angle_mother - sign * angle_daughter2)
        
        
        if daughter_1_x > max_x:
            max_x = daughter_1_x
        if daughter_2_x > max_x:
            max_x = daughter_2_x
        if daughter_1_y > max_y:
            max_y = daughter_1_y
        if daughter_2_y > max_y:
            max_y = daughter_2_y
        if daughter_1_y < min_y:
            min_y = daughter_1_y
        if daughter_2_y < min_y:
            min_y = daughter_2_y
        
        dict_node_property[mother_id] = (dict_node_property[mother_id][0], # position unchanged
                                         [edge_counter, edge_counter+1],   # added the edges
                                         proba_list[branching_count])      # added the probability of that node

        dict_node_property[true_daughter1_id] = ((daughter_1_x, daughter_1_y ), [], "EX")
        dict_node_property[true_daughter2_id] = ((daughter_2_x, daughter_2_y ), [], "EX")
        dict_edge_property[edge_counter+1] = ((mother_id, true_daughter2_id), particle_info[true_daughter2_id][0])
        dict_edge_property[edge_counter]   = ((mother_id, true_daughter1_id), particle_info[true_daughter1_id][0])
        dict_angles_of_part[true_daughter1_id] = angle_mother + sign * angle_daughter1
        dict_angles_of_part[true_daughter2_id] = angle_mother - sign * angle_daughter2
        if dict_memorise_mother_direction[mother_id] == 0:
            dict_memorise_mother_direction[true_daughter1_id] = 0
            dict_memorise_mother_direction[true_daughter2_id] = 0
        else:
            dict_memorise_mother_direction[true_daughter1_id] = 0
            dict_memorise_mother_direction[true_daughter2_id] = 0

        edge_counter += 2
        # Note that the final nodes will have no edge and 0 proba.
            
    # Now: plot the nodes in dict_node_property with edges listed in dict_edge_property
    
    figure = draw_tree(dict_node_property, dict_edge_property, (label, total_proba), "JUNIPR tree", (max_x, max_y, min_y), path, max_value_colour =energy_max, return_plt_bool = return_plt_bool)
    return figure



