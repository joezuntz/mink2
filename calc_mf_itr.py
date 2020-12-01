import os
import numpy as np
from mf import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]

# path for given inputs

path = '/disk01/ngrewal/Fiducial_Simulations'
path_mf = '/disk01/ngrewal/MFs'

# define variables
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
thr_ct = 10

# find MFs for all iterations 
# load maps
cmaps = np.load(os.path.join(path, f'cmaps_{index}.npy'))  # clustering maps
lmaps = np.load(os.path.join(path, f'lmaps_{index}.npy'))  # lensing maps

# calculate MFs
v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct)
v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
    
# save MFs
np.save(os.path.join(path_mf, f'V_{index}'),v_all)
