import os
import numpy as np
import matplotlib.pyplot as plt

# locations of MFs and Cls
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# structure variables
itr = 1000                                                     # totlal simulations instructed in run_sim.py (not all of these will run bc of cuillin hardware failures)
cl_len = len(np.load(os.path.join(cl_path,'C_1.npy'))[0])      # import length of Cls because this changes elsewhere
map_count = 270 + 9*cl_len                                     # 9 maps * threshold count * 3MFs + 9 * Cls
v_all = np.zeros((itr,map_count))                              # use indexing
e = 0                                                          # count number of errors

for i in range(itr):
    try:
       v = np.load(os.path.join(mf_path,f'V_{i+1}.npy'))       # load MFs
       c = np.load(os.path.join(cl_path,f'C_{i+1}.npy'))       # load Cls
       v_all[i] = np.concatenate((v,c.flatten()),axis=None)    # concatenate MFs and Cls
    except:
        e += 1

V  = v_all[~(v_all==0).all(1)]                                 # remove rows with zeros (where the simulation failed)
    
np.save('v_all',V)                                             # save concatenated array in current directory
print('errors=',e)                                             # show number of simulations that failed
