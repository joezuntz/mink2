import os
import numpy as np
import healpy as hp
import math
from mf import *
from cl import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]

# define inputs
nside = 1024
smoothing_arcmin = 5
thr_ct = 10
sky_frac = 1
m_type = 'c+l' # c+l,c,l

# path for given inputs
path = '/disk01/ngrewal/Fiducial_Simulations'
path_mf = '/disk01/ngrewal/MFs'
path_cl = '/disk01/ngrewal/Cls'

# define index of simulation and output
index = int(os.environ['SLURM_ARRAY_TASK_ID'])

# load maps
cmaps = np.load(os.path.join(path, f'cmaps_{index}.npy'))  # clustering maps
lmaps = np.load(os.path.join(path, f'lmaps_{index}.npy'))  # lensing maps

 
## Simplify Maps ##

# reduce pixel size - don't need to do bc im using nside=1024 in all tests
#cmaps1 = hp.ud_grade(cmaps,nside,power=-2)         
#lmaps1 = hp.ud_grade(lmaps,nside)
cmaps1 = cmaps
lmaps1 = lmaps

# smooth maps
smoothing = np.radians(smoothing_arcmin/60)      # convert arcmin to degrees
cmaps2 = np.zeros((len(cmaps1),12*nside**2))   
lmaps2 = np.zeros((len(lmaps1),12*nside**2))

for i in range(len(cmaps1)):
    cmaps2[i] = hp.smoothing(cmaps1[i],fwhm=smoothing)
 
for i in range(len(lmaps1)):
    lmaps2[i] = hp.smoothing(lmaps1[i],fwhm=smoothing) 

    
## Analysis ##    
f = int(math.floor(sky_frac*12*nside**2))

# calculate MFs
# clustering and lensing
if m_type=='c+l':
    v,v0,v1,v2 = calc_mf_2maps(cmaps2,lmaps2,thr_ct,f)
    c = Cl_2maps(cmaps2,lmaps2,nside,f).flatten()

# clustering
if m_type=='c':
    v,v0,v1,v2 = calc_mf_2maps(cmaps2,[],thr_ct,f)
    c = Cl_2maps(cmaps2,[],nside,f).flatten()

# lensing
if m_type=='l':
    v,v0,v1,v2 = calc_mf_2maps([],lmaps2,thr_ct,f)
    c = Cl_2maps([],lmaps2,nside,f).flatten()

v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))

# save MFs and Cls
np.save(os.path.join(path_mf, f'V_{index}_s{smoothing_arcmin}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}'),v_all)
np.save(os.path.join(path_cl, f'C_{index}_s{smoothing_arcmin}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}'),c)
