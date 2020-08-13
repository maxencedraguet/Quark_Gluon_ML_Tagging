 #############################################################################
#
# JuniprProcessing.py
#
# A list of function to be employed for processing a constituent and a jet
# dataset into a junipr ready dictionnary.
#
# The algorithm employed in clustering history is the antiKT using PyJet
#
# Author -- Maxence Draguet (30/06/2020)
#
#############################################################################

import math
import numpy as np
import pandas as pd
from pyjet import cluster
import json

def get_4mom(jet):
    """
    Returns a list with the info ['E', 'px', 'py', 'pz'] from a PseudoJet object.
    """
    return [float("{:.8f}".format(jet.e)), float("{:.8f}".format(jet.px)), float("{:.8f}".format(jet.py)), float("{:.8f}".format(jet.pz))]

def get_3mom_angles_tuple(jet):
    """
    Returns a list with the info ['E', 'eta', 'phi'] from a PseudoJet object.
    """
    return tuple([float("{:.8f}".format(jet.e)), float("{:.8f}".format(jet.eta)), float("{:.8f}".format(jet.phi))])


def get_4mom_from_tuple(jet):
    """
    Returns a tuple with float of .8 from a regular tuple.
    """
    return tuple([float("{:.8f}".format(jet[0])), float("{:.8f}".format(jet[1])), float("{:.8f}".format(jet[2])), float("{:.8f}".format(jet[3]))])


def get_4mom_EM_rel(jet, rel):
    """
    Returns a list with the info ['E', 'theta', 'phi', 'mass'] from a PseudoJet object jet relative to a PseudoJet object rel.
    """
    jet_vec = [jet.px, jet.py, jet.pz]
    rel_vec = [rel.px, rel.py, rel.pz]
    energy = jet.e
    mass   = jet.mass
    if mass < 0: # in some cases, one of the daughter mass was negative
        #print("Case of negative mass: ", mass)
        mass = 0
    theta  = get_angle_between_momenta(jet_vec, rel_vec)
    phi    = get_plane_of_momenta(jet_vec, rel_vec)
    return [float("{:.8f}".format(energy)), float("{:.8f}".format(theta)), float("{:.8f}".format(phi)), float("{:.8f}".format(mass))]

def find_softest(list_p):
    """
    From a pair of pseudo jet in the list list_p, return p1, p2 with p1 being the softest (less energy) constituent
    """
    pA, pB = list_p
    if pA.e < pB.e:
        return pA, pB
    return pB, pA

def factorise_branching(mother, dh1, dh2):
    """
    Inputs are the mother and two daughters pseudojets with dh1 being the soft daughter.
    From this info, computes and returns the relevant branching variable to form the 4-mom
    [z, theta, phi, delta]
    Where:
    - z is the fraction of energy of the soft daughter on the mother
    - theta is the angle, in the plane of the branching, between the mother and the soft daughter
    - phi is the angle of the plane of branching
    - delta is the angle between the mother and the hard daughter
    """
    mom_vec = [mother.px, mother.py, mother.pz]
    dh1_vec  = [dh1.px, dh1.py, dh1.pz]
    dh2_vec  = [dh2.px, dh2.py, dh2.pz]
    info_z     = dh1.e / mother.e
    info_theta = get_angle_between_momenta(dh1_vec, mom_vec)
    info_delta = get_angle_between_momenta(dh2_vec, mom_vec)
    info_phi   = get_plane_of_momenta(dh1_vec, mom_vec)
    return [float("{:.8f}".format(info_z)), float("{:.8f}".format(info_theta)), float("{:.8f}".format(info_phi)), float("{:.8f}".format(info_delta))]


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
    return phi_x


def perform_clustering(cluster_algorithm, cluster_radius, jet_pdf, constituent_pdf):
    """
    This routines applies the specified cluster_algorithm to the constituents of the jet (with the specified cluster_radius).
    
    The output is a dictonnary matching the format of JUNIPR that can be directly saved to JSON.
    
    Need:
    
    - label: (classify the sort of jet)
    - multiplicity: number of final state particle
    - n_branchings: number of branching = multiplicty -1
    note: there are a total of mutliplicity + n_branching set of momenta
    - seed_momentum: the mother of all mother info
    - CSJets: set of all momenta ... the indices in this list are used in other places
    - CS_ID_intermediate_states:
    a list of list. In each sublist, lists the indices of particles at the moment considered
    There is a sublist for each state of the clustering (for 1 to x particles, there should be x lists so there are as many sublists as the multiplicty)
    - mother_id_energy_order: from the intermediate state particles ranked by energy, indicate which one is the mother (also among final state particle)
    - CS_ID_mothers: the ID's of the mother that decay, from first to decay to the last,
    - CS_ID_daugthers: list of list. Sublists are pairs of daughters produced by the decay of one of the mother. Some of these might be mothers later
    - branching: this a a peculiar list of lists. The subist collect 4 numbers representing the branching.
    
    - mother_momenta: list of mother momenta (each 4 momenta presented as a list).
    - daugther_momenta: mist of daugther momentas. Pairs of daughters are in the sublists, with each momenta of daugther being a list.
    
    Note: a confusing element in the PyJet algorithm.
    They call "mother" the particles that are clustered, not the clustered particle. This effectively inverts the role of mother-daughter compared to that
    discussed in a branching. Where one typically says that ONE mother decays to TWO daughters, PyJet says that TWO mothers combined in ONE daughter. The naming
    defined here reflects the typical version, only the PyJet function "parents" and "child" adopt the PyJet way (parents = daughters and child = mother).
    """
    if cluster_algorithm == 'cambridge':
        algo_p_value = 0
    elif cluster_algorithm == 'antikt':
         algo_p_value = -1
    else:
        raise ValueError("Cluster algorithm {} not recognised". format(cluster_algorithm))

    DTYPE_EP = np.dtype('f8,f8,f8,f8')
    cpdf = constituent_pdf.copy(deep = True) # .iloc[:100,:]
    
    cpdf['4-momentum'] = list(zip(cpdf['constituentE'], cpdf['constituentPx'], cpdf['constituentPy'], cpdf['constituentPz']))
    cpdf = cpdf[['entry', 'constituentJet', '4-momentum', 'isTruthQuark']]
    cpdf = cpdf.groupby(['entry', 'constituentJet', 'isTruthQuark'])['4-momentum'].apply(list).reset_index(name='4-mom_list') ## There is a faster way of doing this with numpy
    
    #cpdf = pd.merge(cpdf, j_pdf, how='left', left_on=['entry', 'constituentJet'], right_on=['entry', 'subentry'])
    # the dataset is now trimmed: only got the index and the list of 4-momenta constituents. The next step would be to cluster these.
    
    # One way to cluster is to use the python implementation of fastjest : pyjet.
    column_data = cpdf['4-mom_list'].to_numpy() #only two for now [:2]
    label_data  = cpdf['isTruthQuark'].to_numpy() #only two for now [:2]
    
    collected_data = dict()
    collected_exceptions = dict()
    junipr_ready_datapoint = list()
    list_of_exceptional_events = list()
    #for filou, elem in enumerate(tqdm(column_data)):
    for filou, elem in enumerate(column_data):
        #print("\n JET {}\n".format(filou))
        # Initiate the information required for a jet:
        if (filou % 10000 == 0):
            print("Turned about {} events into Junipr Jets".format(filou))
        
        label = label_data[filou]
        multiplicity = len(elem)
        n_branchings = multiplicity - 1
        seed_momentum = list() # The top jet info
        CSJets = [None] * (multiplicity + n_branchings)
        mother_momenta = list()
        daughter_momenta = list()
        CS_ID_intermediate_states = list()
        CS_ID_mothers = list()
        CS_ID_daugthers = list()
        temp_branching_pjet = list()
        branching = list()
        mother_id_energy_order = list()
        
        # Some dictionnary to help with processing
        dic_of_particles_id_mom = dict()
        dic_of_particles_mom_id = dict()

        # Set format for PyJet
        jet_array = np.array(elem,dtype=DTYPE_EP)
        jet_array.dtype.names=['E','px','py','pz']

        # Cluster the constituent into a jet with Radius R and algorithm specified (p = -1).
        # ep = True since momenta are into shape ('E','px','py','pz')
        sequence = cluster(jet_array, algo= cluster_algorithm, R= cluster_radius, ep= True) #p= algo_p_value
        
        jets = sequence.inclusive_jets()  # list of PseudoJets reconstructed from the constituents
        # There should not be more than one reconstructed jet since we're feeding the input of a single jet!
        if (len(jets) !=1):
            print("Warning: from info for a sole jet, {} jets were reconstructed".format(len(jets)))
            #If such a badly reconstructed jet is found, skip it
            number_of_jet_items = len(jets)
            list_of_elements_in_event = dict()
            for counter_jet in range(number_of_jet_items):
                the_jet = jets[counter_jet]
                list_of_constituents = list()
                for ind, consti in enumerate(the_jet):
                    constituent = get_3mom_angles_tuple(consti)
                    list_of_constituents.append(constituent)
                list_of_elements_in_event[counter_jet] = [get_3mom_angles_tuple(the_jet), list_of_constituents]
        
            list_of_exceptional_events.append({"event_count":filou, "number_component":number_of_jet_items, "dictionnary_component":list_of_elements_in_event})
                #if len(list_of_exceptional_events) == 20: #WARNING: abnormal use
                #    break
            continue
                #continue #WARNING: abnormal use
        jet = jets[0] #jet is the first and sole element of this list
        #print("The inclusive jet is ",jet)
        seed_momentum = get_4mom_EM_rel(jet, jet)
        #print("Jet info global = ", seed_momentum)
            
        # Let's recuperate the final states particles: those we fed.
        # Note that the stored information is relative to the top jet! This is not the absolute
        # info gathered by the simulations
        for ind, consti in enumerate(jet):
            #print("Elem {}: {}".format(ind, consti))
            #print("Elem {} relative: {}".format(ind, get_4mom_EM_rel(consti, jet)))
            constituent = get_4mom_EM_rel(consti, jet)
            CSJets[ind] = constituent
            dic_of_particles_id_mom[ind] = constituent
            dic_of_particles_mom_id[tuple(constituent)] = ind

        """
        for ind, consti in enumerate(jet.constituents_array()):#ep=True)):
            print("Elem {}: {}\n".format(ind, consti))
            CSJets[ind] = list(get_4mom_from_tuple(consti))
            dic_of_particles_id_mom[ind] = list(get_4mom_from_tuple(consti))
            print(get_4mom_from_tuple(consti))
            dic_of_particles_mom_id[get_4mom_from_tuple(consti)] = ind
        """
        """
        print("CSJETS before the new \n",)
        for ind, elem in enumerate(CSJets):
        print("{}, {}\n".format( ind, elem))
        """
        particles_available = list(jet)

        # This is the state of the jet through each step of the algorithm. Will be translated into jet indices later.
        # For now its bottom-up (final state particles to mother particle). This will be inverted after the while loop.
        CS_ID_intermediate_states.append([tuple(get_4mom_EM_rel(subjet, jet)) for subjet in particles_available])

        possible_mothers = []          # tracks mothers available for selection a step (both daughters are in intermediate state)
        particles_step = particles_available.copy() # tracks available daughter at step that haven't yet been matched to a mother
        number_branching = 0
        location_First_None = multiplicity #the first avilable value in CSJet (the one after the final state particles).
        
        cataclysm_bool = False # set to true if you see something fishy. One of the error observed is that the output of elem.child is nonetype.
        # If this happens, just get rid of everything.
        while (number_branching < n_branchings):
            next_particles_step = list()
            observed_candidate_mothers = []
            for elem in particles_step:
                mother_part = elem.child # remember the note from function definition: PyJet child is in fact the mother.
                if mother_part is None:
                    # this is a cataclysmic event: makes no sense to consider a particle with no mother here. The only one that it could be is the top mother but it shouldn't be in the loop.
                    # This is very rare so get rid of it
                    cataclysm_bool = True
                if cataclysm_bool:
                    last_elem = elem
                    break
                if mother_part in observed_candidate_mothers:
                    # Mother has already been observed (you stumbled upon the second daughter)
                    continue
                
                daughter1, daughter2 = mother_part.parents
                if not(daughter1 in(particles_step) and daughter2 in(particles_step)):
                    # One of the daughter is not yet in the list, reconsider the case later when both are there
                    next_particles_step.append(elem)
  
                else:
                    # Case where both daughters are there: add mother to potential list of next particle
                    deltaR2 = (daughter1.eta - daughter2.eta)**2 + (daughter1.phi - daughter2.phi)**2
                    possible_mothers.append((deltaR2, mother_part))
                    observed_candidate_mothers.append(mother_part)
            if cataclysm_bool:
                break
            # Select the mother in the branching where deltaR**2 is minimised between components
            # Important: note that is also works for anti-kT as there will be only one possible mother (since a core particle gets all soft radiations added to it).
            min_deltaR2, selected_mother = min(possible_mothers)
            possible_mothers.remove((min_deltaR2, selected_mother))    # the selected mother should no longer be considered for next iteration
            step_daughter1, step_daughter2 = selected_mother.parents   # Retrieve the selected mother's daughters

            # Add the selected mother to information gathered
            selected_mother_info = get_4mom_EM_rel(selected_mother, jet)
            CSJets[location_First_None] = selected_mother_info
            dic_of_particles_id_mom[location_First_None] = selected_mother_info
            dic_of_particles_mom_id[tuple(selected_mother_info)] = location_First_None

            # Add info on branching (branching, mother and daughters)
            temp_branching_pjet.append([selected_mother ,[step_daughter1, step_daughter2] ])
            mother_momenta.append(selected_mother_info)
            daughter_momenta.append([get_4mom_EM_rel(step_daughter1, jet), get_4mom_EM_rel(step_daughter2, jet)])

            # Update and store intermediate state
            particles_available.remove(step_daughter1)
            particles_available.remove(step_daughter2)
            particles_available.append(selected_mother)
            CS_ID_intermediate_states.append([tuple(get_4mom_EM_rel(subjet, jet)) for subjet in particles_available])

            # Prepare next iteration
            next_particles_step.append(selected_mother)
            particles_step= next_particles_step
            number_branching += 1
            location_First_None += 1
                
            #print("N_branching currently = {}, total = {}".format(number_branching, n_branchings))
        # End of "while"
        if cataclysm_bool:
            print("A cataclysmic error occured: at some point, a mother was None. It is the mother of {}, for a jet starting with {}.".format(last_elem, jet))
            # Terrible error occured: a Nonetype mother. Strange error that seems very rare. Just discard the whole event
            continue
        # Reverse the list created along the above iteration.
        # This is required has the iteration was bottom-up (starting with final state particles, go back to mother one) while JUNIPR starts with top-mother
        CS_ID_intermediate_states.reverse()
        mother_momenta.reverse()
        daughter_momenta.reverse()
        temp_branching_pjet.reverse()

        """
        list_elem = [jet]
        list_elem_updated = []
        stock_new_elem = []
        location_last_None = multiplicity + n_branchings - 1 # Track this one to add the next decaying intermediary particle at the last of the None
        while (len(list_elem) < multiplicity):
            #print("Before for ", list_elem)
            for elem in list_elem:
                if elem.parents:
                    # A decaying element, should be added to the list of momenta
                    CSJets[location_last_None] = get_4mom_EM_rel(elem, jet)
                    dic_of_particles_id_mom[location_last_None] = get_4mom_EM_rel(elem, jet)
                    dic_of_particles_mom_id[tuple(get_4mom_EM_rel(elem, jet))] = location_last_None
                    location_last_None -= 1 #the last none is now a place behind
                    #print("Elem {} has parents\n".format(elem))
                    sub_jet1, sub_jet2 = elem.parents
                    list_elem_updated.extend([sub_jet1, sub_jet2])
                    # stock new intermediary particles: does that have parents
                    # (otherwise they're final state and we already have them).
                    print("Particle 1 daughter: ", sub_jet1.child)
                    print("Particle 2 daughter: ", sub_jet2.child)
                    if sub_jet1.parents:
                        stock_new_elem.append(sub_jet1)
                    if sub_jet2.parents:
                        stock_new_elem.append(sub_jet2)
                    mother_momenta.append(get_4mom_EM_rel(elem, jet))
                    temp_branching_pjet.append([elem ,[sub_jet1,  sub_jet2] ])
                    daughter_momenta.append([get_4mom_EM_rel(sub_jet1, jet), get_4mom_EM_rel(sub_jet2, jet)])
                    
                    print("Subjet1 ",sub_jet1)
                    print("Subjet2 ",sub_jet2.constituents_array(ep=True))
                    print("Subjet1 ", get_4mom(sub_jet1))
            
                else:
                    list_elem_updated.append(elem)
            # end "for"
            list_elem = list_elem_updated
            CS_ID_intermediate_states.append([tuple(get_4mom_EM_rel(subjet, jet)) for subjet in list_elem_updated])
            list_elem_updated = []
        # end "while"
        """
        """
        print("CSJETS after the new \n")
        for ind, elem in enumerate(CSJets):
            print("{}, {}\n".format( ind, elem))
        """
        """
        print("The new elements uncovered \n")
        for ind, elem in enumerate(stock_new_elem):
            print("{}, {}\n".format( ind, get_4mom(elem)))
        """
        """
        # To check that the final elements are indeed the inital particles
        for elem in list_elem:
            print("Does elem {} has parents? {}".format(elem, elem.parents))
        """
        """
        print("The dictionnary\n")
        for key in dic_of_particles_mom_id:
            print("{}, {}\n".format(key, dic_of_particles_mom_id[key]))
        """
        """
        # Checking the mother and daughters momenta list.
        print("\nChecking mother and daughters\n")
        for count, elem in enumerate(mother_momenta):
            print("Indices {}, mother {} and daugthers {} {}".format(count, dic_of_particles_mom_id[tuple(elem)], dic_of_particles_mom_id[tuple(daughter_momenta[count][0])], dic_of_particles_mom_id[tuple(daughter_momenta[count][1])]))
            print("Momenta {}, mother {} and daugthers {}".format(count, elem, daughter_momenta[count]))
        """
        """
        # Short test to verify dictionnary access with tuple of float keys.
        if filou == 1:
            print("For entry (67.73412037, 27.13691521, -61.0981493, 10.62593341): ",dic_of_particles_mom_id[(67.73412037, 27.13691521, -61.0981493, 10.62593341)])
            print("For entry (1.19000721, 0.62274963, -0.98128253, 0.25570434): ",dic_of_particles_mom_id[(1.19000721, 0.62274963, -0.98128253, 0.25570434)])
        """
            
        """
        print("\nChecking jet state\n")
        for cou, entry in enumerate(CS_ID_intermediate_states):
            print("{}, {}".format(cou, entry))
        """
        # Convert some of the momenta list into indices lists
        
        # Note that CS_ID_intermediate_states_ind is ordered by energy (easier for mother_id_energy_order)
        CS_ID_intermediate_states_ind = list()
        for elem in CS_ID_intermediate_states:
            # order the list by energy (the 0 component of each subelement in elem
            elem_sorted = sorted(elem, key=lambda x: x[0], reverse = True)
            new_format = list()
            for subelem in elem_sorted:
                # subelem is a 4-momenta that is here converted to an indice
                new_format.append(dic_of_particles_mom_id[subelem])
            CS_ID_intermediate_states_ind.append(new_format)

        CS_ID_mothers = list()
        CS_ID_daugthers = list()
        for elem in mother_momenta:
            CS_ID_mothers.append(dic_of_particles_mom_id[tuple(elem)])
        for elem in daughter_momenta:
            CS_ID_daugthers.append([dic_of_particles_mom_id[tuple(elem[0])], dic_of_particles_mom_id[tuple(elem[1])]  ])

        """
        print("\nChecking jet state with indices\n")
        for cou, entry in enumerate(CS_ID_intermediate_states_ind):
            print("{}, {}".format(cou, entry))

        print("\nChecking mother id\n")
        for elem in CS_ID_mothers:
            print("{}\n".format(elem))
        print("\nChecking daughter id\n")
        for elem in CS_ID_daugthers:
            print("{}\n".format(elem))
        """

        # With the info just gathered, write list of mother_id_energy_order

        mother_id_energy_order.append(0)
        for counter, present_mother in enumerate(CS_ID_mothers):
            # present_mother is the one decaying at the row considered.
            if counter == 0:
                # a first there can only be one mother so it's already in the list
                continue
            # CS_ID_intermediate_states_ind is ordered by energy: the mother index in this list is her ranking
            mother_id = CS_ID_intermediate_states_ind[counter].index(present_mother)
            mother_id_energy_order.append(mother_id)
        """
        print("\nChecking mother_energy_order\n")
        for cou, elem in enumerate(mother_id_energy_order):
            print("{}, {}\n".format(cou, elem))
        """

        # Now the branching info. Start with temp_branching_pjet
        # dataformat is [ [mother, [daughter1, daughter 2] , ... ]
        # Warning: all of these are still in absolute coordinate. I believe this is normal
        # (since geometrical info is relative to the mother, so does not matter what the whole jet does.
        for cou, elem in enumerate(temp_branching_pjet):
            mother = elem[0]
            dh1, dh2 = find_softest(elem[1])
            branching_info = factorise_branching(mother, dh1, dh2)
            branching.append(branching_info)
        """
        print("\nChecking branching\n")
        for cou, elem in enumerate(branching):
            print("{}, {}\n".format(cou, elem))
        """

        # Now make a dictionnary with these entry and add it to the junipr_ready_datapoint list
        datapoint = dict()
        datapoint["label"]  = int(label)
        datapoint["multiplicity"]  = multiplicity # useless
        datapoint["n_branchings"]  = n_branchings
        datapoint["seed_momentum"] = seed_momentum
        datapoint["CSJets"]        = CSJets
        datapoint["CS_ID_intermediate_states_ind"] = CS_ID_intermediate_states_ind  # useless
        datapoint["mother_id_energy_order"] = mother_id_energy_order
        datapoint["CS_ID_mothers"]   = CS_ID_mothers
        datapoint["CS_ID_daugthers"] = CS_ID_daugthers
        datapoint["branching"] = branching
        datapoint["mother_momenta"]   = mother_momenta
        datapoint["daughter_momenta"] = daughter_momenta

        junipr_ready_datapoint.append(datapoint)
        
    collected_data["JuniprJets"] = junipr_ready_datapoint
    collected_exceptions["JuniprException"] = list_of_exceptional_events
    return collected_data, collected_exceptions








