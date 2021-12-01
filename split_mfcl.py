import os
import numpy as np
import subprocess
from astropy.io import fits

path = '/home/ngrewal/mink2/simulation_code/new_data/'
dirname = os.path.dirname(path)

# get parameter values to load relevant file - these will be sys inputs
smoothing = 5
nside = 1024
thr_ct = 10
sky_frac = 0.44
seed = 10291995
a_type = 'MF+Cl' # ALWAYS
m_type = 'c+l'
source = 'lsst_y1'
source_file = os.path.abspath(os.path.join(dirname, source+".fits"))

# load source file to get redshift and galaxy bias bin counts
f = fits.open(source_file)
nsource = f["SOURCE"].header['NBIN']
nlens = f["LENS"].header['NBIN']
f.close()

# get relevant bin count                                                                            
if m_type == 'c+l':
    nmap = nsource + nlens
elif m_type == 'l':
    nmap = nsource
elif m_type == 'c':
    nmap = nlens
else:
    raise ValueError(f"Unknown m_type {m_type}")

# total length of observables
nmf = nmap * thr_ct * 3

# load MF+Cl array
obs = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.npy')

# split array
mf = obs[:,:nmf]
cl = obs[:,nmf:]
print(' MF shape: ',mf.shape,'\n Cl shape: ',cl.shape,'\n total shape: ',obs.shape)

# save array
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_MF_{m_type}_{source}.npy',mf)
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_Cl_{m_type}_{source}.npy',cl)
