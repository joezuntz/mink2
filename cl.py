import pymaster as nmt
import numpy as np

 
# calculate Cl for any map with masking applied
def Cl_func(map_,b):
    mask = np.ones(len(map_))                   # build mask
    f_0 = nmt.NmtField(mask,[map_])             # initialise spin-0
    cl_00 = nmt.compute_full_master(f_0,f_0,b)  # computer MASTER estimator
    return cl_00[0]

# calculate Cls for clustering and lensing maps
def Cl_2maps(c_map,l_map,nside):

    b = nmt.NmtBin.from_nside_linear(nside,50)   # apply binning with 4 ells per bandpower 
    cl_len = b.get_n_bands()                    # length of Cls
    map_len = 9                                 # number of clustering and lensing maps
    cl = np.zeros((map_len,cl_len))
    for i in range(len(c_map)):
        cl[i] = Cl_func(c_map[i],b)
    for j in range(len(l_map)):
        cl[i+j+1] = Cl_func(l_map[j],b)
    return cl
