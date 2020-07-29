import numpy as np
from coefficient import * 

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

def setup_threshold_mf(itr,nside,smoothing,b):
    
    N = 12*nside*nside 
    
    # create fixed maps with iterations
    c_map_fixed = np.zeros((itr,5,N)) 
    l_map_fixed = np.zeros((itr,4,N))

    for i in range(itr):
        c_map_fixed[i], l_map_fixed[i] = simulate_des_maps(0.3, 0.8, smoothing, nside)
        print(i)
        
    # points (get constants from plotting notebook)
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = 0.8989639361571576*omega_m + 0.5303108191528527

    S_8 = np.zeros(b)
    for i in range(b):
        S_8[i] = sigma_8[i] * (omega_m[i]/0.3)**0.5329788249790618  
        
    # create maps with changing omega and sigma
    c_map = np.zeros((b,5,N)) 
    l_map = np.zeros((b,4,N))

    for i in range(b):
        c_map[i], l_map[i] = simulate_des_maps(omega_m[i], sigma_8[i], smoothing, nside)
        print(i)
             
    return c_map_fixed,l_map_fixed,c_map,l_map,S_8


def threshold_mf(thr_ct,itr,c_map_fixed,l_map_fixed,c_map,l_map,S_8,nside,map_len,b):
    
    array_len = map_len*thr_ct*3
    
    N = 12*nside*nside  

    ## fixed maps
    v_all_fixed = np.zeros((itr,array_len)) 
    for i in range(itr):
        v_fixed, v0_fixed, v1_fixed, v2_fixed = calc_mf_2maps(c_map_fixed[i],l_map_fixed[i],thr_ct,N)
        v_all_fixed[i] = np.concatenate((v0_fixed.flatten(),v1_fixed.flatten(),v2_fixed.flatten()))
        print(i)

    # stack 100 iteration fixed variable versions of V0,V1,V2
    v_all_mean = np.zeros(array_len)
    for i in range(array_len):
        v_all_mean[i] = mean(v_all_fixed[:,i])

    ## changing maps
    V_all = np.zeros((b,array_len))
    for i in range(b):
        v, v0, v1, v2 = calc_mf_2maps(c_map[i],l_map[i],thr_ct,N)
        V_all[i] = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        print(i)    
       
    # covariance
    cov = np.cov((v_all_fixed.transpose()))

    # singular matrix workaround - do we only want positive covariance values or just not = 0?
    good = cov.diagonal() > 0
    cov2 = cov[good][:, good]

    # calculate the likelihood          
    L = np.zeros(b)

    try:
        inv_cov = np.linalg.inv(cov)
        for i in range(b):
            L[i] = -0.5 * (V_all[i] - v_all_mean) @ inv_cov @ (V_all[i] - v_all_mean)
    except:
        inv_cov2 = np.linalg.inv(cov2)
        for i in range(b):
            d = (V_all[i] - v_all_mean)[good]
            L[i] = -0.5 * d @ inv_cov2 @ d

    coefficient = np.polyfit(S_8,L,2)
    return np.sqrt(-1 / (2*coefficient[0]))