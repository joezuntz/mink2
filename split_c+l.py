import numpy as np
import pymaster as nmt
import os
from astropy.io import fits

# define inputs
sky_frac = 0.44
a_type = 'Cl'
smoothing = 5
nside = 1024
thr_ct = 10
source = 'lsst_y1'
source_file = os.path.abspath(os.path.join(os.getcwd(), "simulation_code/new_data", source+".fits"))

# load source file to get bin count
f = fits.open(source_file)
l_bins = f["SOURCE"].header['NBIN']
c_bins = f["LENS"].header['NBIN']
f.close()

# doesn't change
m_type = 'c+l'

# load all array
a = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.npy')


if a_type == 'MF':
    c = c_bins*thr_ct
    l = l_bins*thr_ct
    clust = np.concatenate((a[:,0:c],a[:,c+l:c+l+c],a[:,c+l+c+l:c+l+c+l+c]),axis=1)
    lens = np.concatenate((a[:,c:c+l],a[:,c+l+c:c+l+c+l],a[:,c+l+c+l+c:]),axis=1)
    
# get Cl info
bins = nmt.NmtBin.from_lmax_linear(lmax=int(1.5*nside),nlb=50)
cl_len = bins.get_n_bands()

if a_type == 'Cl':
    m = c_bins*cl_len # Cls are all clustering then all lensing data points 
    clust = a[:,0:m]  # m = midpoint
    lens = a[:,m:]
    
if a_type == 'MF+Cl': # saved MFs first then Cls in array
    
    # get MFs
    c = c_bins*thr_ct
    l = l_bins*thr_ct
    clust_mf = np.concatenate((a[:,0:c],a[:,c+l:c+l+c],a[:,c+l+c+l:c+l+c+l+c]),axis=1)
    lens_mf = np.concatenate((a[:,c:c+l],a[:,c+l+c:c+l+c+l],a[:,c+l+c+l+c:c+l+c+l+c+l]),axis=1)
      
    # get Cls 
    f = c+l+c+l+c+l # last MF index
    m = c_bins*cl_len
    clust_cl = a[:,f:f+m]  # m = midpoint
    lens_cl = a[:,f+m:]
    
    # concatenate by map type
    clust = np.concatenate((clust_mf,clust_cl),axis=1)
    lens = np.concatenate((lens_mf,lens_cl),axis=1)

    
print(a_type,' Total: ',a.shape,'/n Clustering: ',clust.shape,'/n Lensing: ',lens.shape)

# save array
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_c_{source}.npy',clust)
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_l_{source}.npy',lens)