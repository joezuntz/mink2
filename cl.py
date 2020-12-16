import pymaster as nmt
import numpy as np

 
def Cl_func(map_,mask,b):
    
    """
    Parameters
    ----------
    map_ : clustering or lensing map
    mask : mask applied to map (same array of ones used in each instace)
    b : number of ells per bandpower

    Returns
    -------
    cl_00[0] : Cl value for the given map
    """
    
    f_0 = nmt.NmtField(mask,[map_])             # initialise spin-0
    cl_00 = nmt.compute_full_master(f_0,f_0,b)  # computer MASTER estimator
    return cl_00[0]

# calculate Cls for clustering and lensing maps
def Cl_2maps(c_map,l_map,nside):
    
    """
    Parameters
    ----------
    c_map : 5 clustering maps (1/galaxy bias) bin
    l_map : 4 lensing maps (1/redshift) bin
    nside : number of pixels on each side of the map (total pixel count is 12*nside**2)

    Returns
    -------
    cl : concatenated array of Cls for all 9 clustering and lensing maps
    """

    b = nmt.NmtBin.from_nside_linear(nside,50)   # apply binning with 50 ells per bandpower 
    cl_len = b.get_n_bands()                     # length of Cls
    map_len = 9                                  # number of clustering and lensing maps
    mask = np.ones(12*nside**2)                  # build mask
    
    cl =np.zeros((map_len,cl_len))
    for i in range(len(c_map)):
        cl[i] = Cl_func(c_map[i],mask,b)
    for j in range(len(l_map)):
        cl[i+j+1] = Cl_func(l_map[j],mask,b)
    return cl
