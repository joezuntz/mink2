import os
import numpy as np
import matplotlib.pyplot as plt

# location of MFs
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# concatenate MFs
itr = 1000
map_count = 270 + 9*383  # 9 maps * threshold count * 3MFs
v_all = np.zeros((itr,map_count))  # use indexing

e = 0

for i in range(itr):
    try:
       v = np.load(os.path.join(mf_path,f'V_{i+1}.npy'))
       c = np.load(os.path.join(cl_path,f'C_{i+1}.npy'))
       v_all[i] = np.concatenate((v,c.flatten()),axis=None)
    except:
        e += 1

V  = v_all[~(v_all==0).all(1)]
    
np.save('v_all',V)
print(V,'\n, errors=',e)
