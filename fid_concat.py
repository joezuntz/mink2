import os
import numpy as np
import pymaster as nmt


# vary smoothing, nside, and analysis type
smoothing=5
nside=256
thr_ct = 10
sky_frac = 1
a_type = 'Cl'
m_type = 'l'

# locations of MFs and Cls
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# structure variables
itr = 1000                                                          

# make a list of arrays
if a_type == 'MF':
    
    v = [np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy')) for i in range(itr)]

if a_type == 'Cl':
                                         
    v = [np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy')) for i in range(itr)]

if a_type =='MF+Cl':

    v = [np.concatenate((np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy')),np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))),axis=None) for i in range(itr)]                                                              # concatenate MFs and Cls 


print(len(v))
    
# stack the arrays                                                           
z = np.vstack(v)
print(z.shape)

np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}',z)   # save concatenated array in current directory



'''
# get length of Cls - make sure the ells per bandpower match the number in cl.py
#b = nmt.NmtBin.from_nside_linear(nside,50)
b = nmt.NmtBin.from_lmax_linear(lmax=int(1.5*nside),nlb=50)
cl_len = b.get_n_bands()

# get number of maps
if m_type == 'c':
    map_num = 5
if m_type == 'l':
    map_num = 4
if m_type== 'c+l':
    map_num = 9

    
e = 0                   # count number of errors


# MF + Cl analysis 
if a_type == 'MF+Cl':
    map_count = np.int(map_num*(thr_ct*3 + cl_len))                                                # count of maps * threshold count * 3MFs + 9 maps * Cls
    v_all = np.zeros((itr,map_count))   
    
    for i in range(itr):
        try:
            v = np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load MFs
            c = np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load Cls
            v_all[i] = np.concatenate((v,c),axis=None)                                                                # concatenate MFs and Cls
        except:
            e += 1
    

# MF only 
if a_type == 'MF':
    map_count = np.int(map_num*(thr_ct*3))                                                         # count of maps * threshold count * 3MFs 
    v_all = np.zeros((itr,map_count))   
    
    for i in range(itr):
        try:
            v_all[i] = np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.npy'))    # load MFs
        except:
            e += 1
                                        

# Cl only
if a_type == 'Cl':
    map_count = np.int(map_num*cl_len)                                                           # count of maps * Cls
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
'''
