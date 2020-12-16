import os
import numpy as np
import matplotlib.pyplot as plt

# locations of MFs and Cls
mf_path = '/disk01/ngrewal/MFs'
cl_path = '/disk01/ngrewal/Cls'

# structure variables
itr = 1000                                                     # totlal simulations instructed in run_sim.py (not all of these will run bc of cuillin hardware failures)
cl_len = len(np.load(os.path.join(cl_path,'C_1.npy'))[0])      # import length of Cls because this changes elsewhere
thr_ct = len(np.load(os.path.join(mf_path,'V_1.npy')))/27      # number of threshold counts (divide by map count * MF count)
map_count = np.int(9*(thr_ct*3 + cl_len))                      # 9 maps * threshold count * 3MFs + 9 * Cls
v_all = np.zeros((itr,map_count))                              # use indexing
e = 0                                                          # count number of errors
b = 10                                                         # number of S8 points

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

cov = np.cov(V.transpose())                                    # find the covariance
vc_mean = np.mean(V,axis=0)                                    # find the mean of the fiducial simulation MFs and Cls
i_cov = np.linalg.inv(cov)                                     # find the inverse covariance

# find the likelihood (for now doing this with all fiducial simulations, this should be done for a select amount of non-fiducial simulations)
L = np.zeros(len(V))
for i in range(len(V)):
    diff = V[i] - vc_mean
    L[i] = -0.5 * diff @ i_cov @ diff

np.save('LH',L)
print(len(V))
