#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:03:09 2020

@author: nishagrewal
"""

import pyccl
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from coefficient import *


def lensing(itr,thr_ct,nside,smoothing,b=10):
    
    N = 12*nside*nside
    array_len = thr_ct*3
    
    print("Fixed Map Iterations:")
    
    # fixed map iterations
    v_all_fixed = np.zeros((itr,array_len))
    for i in range(itr):
        c_map, l_map = simulate_des_maps(0.3, 0.8, smoothing, nside, nmax=1)
        v,v0,v1,v2 = calc_mf(l_map[0], thr_ct, N, is_clustering=False)
        v_all_fixed[i] = np.concatenate((v0,v1,v2))
        print(i)
        
    # fixed map mean    
    v_all_mean = np.zeros(array_len)
    for i in range(array_len):
            v_all_mean[i] = np.mean(v_all_fixed[:,i])
            
    # calculate S_8  
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = 0.8989639361571576*omega_m + 0.5303108191528527
    S_8 = sigma_8 * (omega_m/0.3)**0.5329788249790618 
    
    print("MF Calculations:")
    # applying 10 S_8 values
    v_all = np.zeros((b,array_len))
    for i in range(b):
        c_map, l_map = simulate_des_maps(omega_m[i], sigma_8[i], smoothing, nside, nmax=1)
        v,v0,v1,v2 = calc_mf(l_map[0], thr_ct, N, is_clustering=False)
        v_all[i] = np.concatenate((v0,v1,v2))  
        print(i)

    # covariance
    cov = np.cov((v_all_fixed.transpose()))
    
    # singular covariance matrix workaround
    good = cov.diagonal() > 0
    cov2 = cov[good][:, good]

    # calculate the likelihood          
    L = np.zeros(b)
    N_ = itr-1           # number of realisations - 1
    p = array_len        # number of data points

    try:
        inv_cov = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov)
        for i in range(b):
            L[i] = -0.5 * (v_all[i] - v_all_mean) @ inv_cov @ (v_all[i] - v_all_mean)
    except:
        inv_cov2 = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov2)
        for i in range(b):
            d = (v_all[i] - v_all_mean)[good]
            L[i] = -0.5 * d @ inv_cov2 @ d
           
            
    coefficient = np.polyfit(S_8,L,2)
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    
    # output array
    c = np.array((thr_ct,smoothing,nside,constraining_power))
     
    ## save data ##

    # parent directory
    output_path = os.path.join(os.getcwd(), '1_Map_Output')

    # save data in new subfolder
    sub_path = os.path.join(output_path, f't{thr_ct}_n{nside}_s{smoothing}')
    

    try:
        os.mkdir(sub_path) 
    except:
        pass

    np.savetxt(os.path.join(sub_path, 'c.out'),c)
    np.savetxt(os.path.join(sub_path, 'V_all_fixed.out'),v_all_fixed)
    np.savetxt(os.path.join(sub_path, 'v_all_changing'),v_all)
