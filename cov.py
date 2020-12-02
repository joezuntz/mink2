import os
import numpy as np
import matplotlib.pyplot as plt

# location of MFs
mf_path = '/disk01/ngrewal/MFs'

# concatenate MFs
itr = 1000
map_count = 270   # 9 maps * threshold count * 3MFs
v_all = np.zeros((itr,map_count))

for i in range(itr):
    try:
        v_all[i] = np.load(os.path.joinmf_(path,f'V_{i+1}.npy'))
    except:
        pass
    
np.save('v_all',v_all)
print(v_all)
