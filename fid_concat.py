import os
import numpy as np

# vary smoothing, nside, and analysis type
smoothing=10
nside=256
thr_ct = 10
sky_frac = 1
a_type = 'MF+Cl'  # 'MF','Cl'
m_type = 'c+l'

# locations of MFs and Cls
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# structure variables
itr = 1000                                                                            # total simulations instructed in run_sim.py (not all of these will run bc of cuillin hardware failures)
cl_len = len(np.load(os.path.join(cl_path,f'C_1_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))[0])      # import length of Cls because this changes elsewhere
thr_ct = len(np.load(os.path.join(mf_path,f'V_1_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy')))/27      # number of threshold counts (divide by map count * MF count)                                                        # use indexing
e = 0                                                                                 # count number of errors
                                                                                 

''' MF + Cl analysis '''
if a_type == 'MF+Cl':
    map_count = np.int(9*(thr_ct*3 + cl_len))                                                # 9 maps * threshold count * 3MFs + 9 maps * Cls
    v_all = np.zeros((itr,map_count))   
    
    for i in range(itr):
        try:
           v = np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load MFs
           c = np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load Cls
           v_all[i] = np.concatenate((v,c.flatten()),axis=None)                                                      # concatenate MFs and Cls
        except:
            e += 1
    

''' MF only '''
if a_type == 'MF':
    map_count = np.int(9*(thr_ct*3))                                                         # 9 maps * threshold count * 3MFs 
    v_all = np.zeros((itr,map_count))   
    
    for i in range(itr):
        try:
           v_all[i] = np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load MFs
        except:
            e += 1
                                        

''' Cl only '''
if a_type == 'Cl':
    map_count = np.int(9*(cl_len))                                                           # 9 maps * Cls
    v_all = np.zeros((itr,map_count))   
    
    for i in range(itr):
        try:
           v_all[i] = np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load Cls
        except:
            e += 1
 
    
 
V  = v_all[~(v_all==0).all(1)]                                                    # remove rows with zeros (where the simulation failed)
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}',V)   # save concatenated array in current directory


# show number of simulations that failed
print('errors=',e)                               
