import os
import numpy as np
import matplotlib.pyplot as plt

# vary smoothing and nside
smoothing=5
nside=512

# locations of MFs and Cls
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# structure variables
itr = 1000                                                                                  # total simulations instructed in run_sim.py (not all of these will run bc of cuillin hardware failures)
cl_len = len(np.load(os.path.join(cl_path,'C_1_s{smoothing_arcmin}_n{nside}.npy'))[0])      # import length of Cls because this changes elsewhere
thr_ct = len(np.load(os.path.join(mf_path,'V_1_s{smoothing_arcmin}_n{nside}.npy')))/27      # number of threshold counts (divide by map count * MF count)
map_count = np.int(9*(thr_ct*3 + cl_len))                                                   # 9 maps * threshold count * 3MFs + 9 * Cls
v_all = np.zeros((itr,map_count))                                                           # use indexing
e = 0                                                                                       # count number of errors

for i in range(itr):
    try:
       v = np.load(os.path.join(mf_path,f'V_{i+1}_s{smoothing_arcmin}_n{nside}.npy'))       # load MFs
       c = np.load(os.path.join(cl_path,f'C_{i+1}_s{smoothing_arcmin}_n{nside}.npy'))       # load Cls
       v_all[i] = np.concatenate((v,c.flatten()),axis=None)    # concatenate MFs and Cls
    except:
        e += 1

V  = v_all[~(v_all==0).all(1)]                                 # remove rows with zeros (where the simulation failed)
np.save('vc_all_s{smoothing}_n{nside}',V)                      # save concatenated array in current directory
print('errors=',e)                                             # show number of simulations that failed
