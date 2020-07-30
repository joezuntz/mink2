#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:43:57 2020

@author: nishagrewal
"""

import numpy as np
import os


def simplified(thr_ct,smoothing,nside,map_len=2,b=10):
        
    array_len = map_len*thr_ct*3
        
    # load data with all 9 maps
    output_path = os.path.join(os.getcwd(), '2_Maps_Output')
    path = os.path.join(output_path, f't{thr_ct}_n{nside}_s{smoothing}')
    v_input_f = np.loadtxt(os.path.join(path, 'v_all_fixed.out'))
    v_input_c = np.loadtxt(os.path.join(path, 'v_all_changing'))
    
    
    ## select columns from first clustering map and first lensing map - positions = 0,5 
    # fixed maps
    v_all_fixed1 = v_input_f[:,0:30]
    v_all_fixed2 = v_input_f[:,150:180]
    v_all_fixed = np.hstack((v_all_fixed1,v_all_fixed2))
    
    # changing maps
    v_all_c1 = v_input_c[:,0:30]
    v_all_c2 = v_input_c[:,150:180]
    v_all_changing = np.hstack((v_all_c1,v_all_c2))
    
    
    # stack 100 iteration fixed variable versions of V0,V1,V2
    v_all_mean = np.zeros(array_len)
    for i in range(array_len):
        v_all_mean[i] = np.mean(v_all_fixed[:,i])
        
    # points (get constants from plotting notebook)
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = 0.8989639361571576*omega_m + 0.5303108191528527

    S_8 = np.zeros(b)
    for i in range(b):
        S_8[i] = sigma_8[i] * (omega_m[i]/0.3)**0.5329788249790618  
        
        
    ## calculate covariance
    cov = np.cov((v_all_fixed.transpose()))  
    
    # singular matrix workaround
    good = cov.diagonal() > 0
    cov2 = cov[good][:, good]

    # calculate the likelihood          
    L = np.zeros(b)

    try:
        inv_cov = np.linalg.inv(cov)
        for i in range(b):
            L[i] = -0.5 * (v_all_changing[i] - v_all_mean) @ inv_cov @ (v_all_changing[i] - v_all_mean)
    except:
        inv_cov2 = np.linalg.inv(cov2)
        for i in range(b):
            d = (v_all_changing[i] - v_all_mean)[good]
            L[i] = -0.5 * d @ inv_cov2 @ d

    # fit a quadratic curve to L(S_8)
    coefficient = np.polyfit(S_8,L,2)
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    print(constraining_power)
    
    # output array
    c = np.array((thr_ct,smoothing,nside,constraining_power))

    # save data in new subfolder
    parent_path = os.path.join(output_path, 'Simplified')
    sub_path = os.path.join(parent_path, f't{thr_ct}_n{nside}_s{smoothing}')

    try:
        os.mkdir(sub_path) 
    except:
        pass

    np.savetxt(os.path.join(sub_path, 'c.out'),c)
    np.savetxt(os.path.join(sub_path, 'V_all_fixed.out'),v_all_fixed)
    np.savetxt(os.path.join(sub_path, 'v_all_changing'),v_all_changing)
    
    
    
def cov_fix(thr_ct,smoothing,nside,map_len=2,b=10,itr=100):
        
    array_len = map_len*thr_ct*3
    
    #load simplified data
    output_path = os.path.join(os.getcwd(), '2_Maps_Output')
    parent_path = os.path.join(output_path, 'Simplified')
    
    # parent folder for input variable combination
    path = os.path.join(parent_path, f't{thr_ct}_n{nside}_s{smoothing}')
        
    v_all_fixed = np.loadtxt(os.path.join(path, 'v_all_fixed.out'))
    v_all_changing = np.loadtxt(os.path.join(path, 'v_all_changing'))
    
    
    # stack 100 iteration fixed variable versions of V0,V1,V2
    v_all_mean = np.zeros(array_len)
    for i in range(array_len):
        v_all_mean[i] = np.mean(v_all_fixed[:,i])
        
    # points (get constants from plotting notebook)
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = 0.8989639361571576*omega_m + 0.5303108191528527

    S_8 = np.zeros(b)
    for i in range(b):
        S_8[i] = sigma_8[i] * (omega_m[i]/0.3)**0.5329788249790618  
        
        
    ## calculate covariance
    cov = np.cov((v_all_fixed.transpose()))  
    
    # singular matrix workaround
    good = cov.diagonal() > 0
    cov2 = cov[good][:, good]

    # calculate the likelihood          
    L = np.zeros(b)
    N_ = itr-1           # number of realisations - 1
    p = array_len        # number of data points
    
    try:
        #inv_cov = np.linalg.inv(cov)
        inv_cov = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov)
        for i in range(b):
            L[i] = -0.5 * (v_all_changing[i] - v_all_mean) @ inv_cov @ (v_all_changing[i] - v_all_mean)
    except:
        #inv_cov2 = np.linalg.inv(cov2)
        inv_cov2 = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov2)
        for i in range(b):
            d = (v_all_changing[i] - v_all_mean)[good]
            L[i] = -0.5 * d @ inv_cov2 @ d

# =============================================================================
#     try:
#         inv_cov = np.linalg.inv(cov)
#         for i in range(b):
#             L[i] = -0.5 * (v_all_changing[i] - v_all_mean) @ inv_cov @ (v_all_changing[i] - v_all_mean)
#     except:
#         inv_cov2 = np.linalg.inv(cov2)
#         for i in range(b):
#             d = (v_all_changing[i] - v_all_mean)[good]
#             L[i] = -0.5 * d @ inv_cov2 @ d
# =============================================================================

    # fit a quadratic curve to L(S_8)
    coefficient = np.polyfit(S_8,L,2)
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    print(constraining_power)
    
    # output array
    c = np.array((thr_ct,smoothing,nside,constraining_power))

    # save data in new subfolder
    parent_path = os.path.join(output_path, 'cov_fix')
    sub_path = os.path.join(parent_path, f't{thr_ct}_n{nside}_s{smoothing}')

    try:
        os.mkdir(sub_path) 
    except:
        pass

    np.savetxt(os.path.join(sub_path, 'c.out'),c)
    np.savetxt(os.path.join(sub_path, 'V_all_fixed.out'),v_all_fixed)
    np.savetxt(os.path.join(sub_path, 'v_all_changing'),v_all_changing)    