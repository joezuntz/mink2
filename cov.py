import os
import numpy as np
import matplotlib.pyplot as plt

# location of MFs
mf_path = '/disk01/ngrewal/MFs'

# concatenate MFs
itr = 1000
map_count = 270   # 9 maps * threshold count * 3MFs
v_all = np.zeros((itr,map_count))

e = 0

for i in range(itr):
    try:
        v_all[i] = np.load(os.path.join(mf_path,f'V_{i+1}.npy'))
    except:
        e += 1
    
np.save('v_all',v_all)
print(v_all,'\n, errors=',e)
