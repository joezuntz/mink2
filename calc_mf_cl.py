import os
import numpy as np
from mf import *
from cl import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]

# path for given inputs

path = '/disk01/ngrewal/Fiducial_Simulations'
path_mf = '/disk01/ngrewal/MFs'
path_cl = '/disk01/ngrewal/CLs'

# define variables
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
thr_ct = 10

# load maps
cmaps = np.load(os.path.join(path, f'cmaps_{index}.npy'))  # clustering maps
lmaps = np.load(os.path.join(path, f'lmaps_{index}.npy'))  # lensing maps

 
## simplify maps

# reduce pixel size
nside = 512
cmaps1 = hp.ud_grade(cmaps,nside)         
lmaps1 = hp.ud_grade(lmaps,nside)

# smooth maps
smoothing = 20
cmaps2 = np.zeros((len(cmaps),12*nside**2))   
lmaps2 = np.zeros((len(lmaps),12*nside**2))

for i in range(len(cmaps)):
    cmaps2[i] = hp.smoothing(cmaps1[i],fwhm=smoothing)
 
for i in range(len(lmaps)):
    lmaps2[i] = hp.smoothing(lmaps1[i],fwhm=smoothing) 

# calculate MFs
v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct)
v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
    
# calculate Cls
c = Cl_2maps(cmaps2,lmaps2,nside)

# save MFs and Cls
np.save(os.path.join(path_mf, f'V_{index}'),v_all)
np.save(os.path.join(path_cl, f'C_{index}'),c)
