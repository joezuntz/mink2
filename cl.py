import pymaster as nmt
import numpy as np


# define an empty dictionary for workspaces of each nside
workspaces = {}

 
def Cl_func(map_,mask,b,w):
    
    """
    Parameters
    ----------
    map_ : clustering or lensing map
    mask : mask applied to map (same array of ones used in each instace)
    b : number of ells per bandpower
    w : workspace corresponding to nside

    Returns
    -------
    cl_00[0] : Cl value for the given map
    """
    
    f_0 = nmt.NmtField(mask,[map_])               # initialise spin-0
    cl_00 = nmt.compute_full_master(f_0,f_0,b,workspace=w)  # computer MASTER estimator using a workspace
    return cl_00[0]


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
    
    
    if nside in workspaces:                              # find the corresponding workspace
        w = workspaces[nside]
    else:                                                # build a workspace for the given nside if it does not exist
        w = nmt.NmtWorkspace()                           # define workspace
        f0 = nmt.NmtField(mask,[np.zeros(12*nside**2)])  # make a field to pass through workspace
        w.compute_coupling_matrix(f0, f0, b)             # compute workspace
        workspaces[nside] = w                            # assign workspace the corresponding value

    
    cl =np.zeros((map_len,cl_len))
    for i in range(len(c_map)):                  # find Cls for clustering maps
        cl[i] = Cl_func(c_map[i],mask,b,w)
    for j in range(len(l_map)):                  # find Cls for lensing maps
        cl[i+j+1] = Cl_func(l_map[j],mask,b,w)
    return cl
