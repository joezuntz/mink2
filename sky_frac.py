#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:27:21 2020

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


def calc_mf_f(m,thr_ct,f,is_clustering):

    v_0 = np.zeros(thr_ct)
    v_1 = np.zeros(thr_ct)
    v_2 = np.zeros(thr_ct)

    
    # THIS NEEDS TO BE FULL SKY MAP
    dm_dtheta, dm_dphi, d2m_dtheta2, d2m_dphi2, d2m_dtheta_dphi = map_derivatives(m)

    std_dev = np.std(m) 
    
    if is_clustering is True:
        v = np.linspace(0,6*std_dev,thr_ct)              # clustering map range
    else:
        v = np.linspace(-3*std_dev, 3*std_dev,thr_ct)    # lensing map range

    m_cut = m[:f]
    #N = f
    
    v_0 = V_0(v, m_cut, f)
    v_1,v_2 = V_12(v, m_cut, dm_dphi, dm_dtheta, d2m_dphi2, d2m_dtheta_dphi, d2m_dtheta2, f)
    
    return v,v_0,v_1,v_2


def calc_mf_2maps_f(clustering_maps,lensing_maps,thr_ct,N):

    map_len=len(clustering_maps)+len(lensing_maps)

    # find MFs for fixed parameter simulation   
    v = np.zeros((map_len,thr_ct))
    v0 = np.zeros((map_len,thr_ct))    
    v1 = np.zeros((map_len,thr_ct))
    v2 = np.zeros((map_len,thr_ct))

    for i,m in enumerate(clustering_maps):
        v[i], v0[i], v1[i], v2[i] = calc_mf_f(m, thr_ct, N, is_clustering=True)

    for j,m in enumerate(lensing_maps):
        v[j+i+1], v0[j+i+1], v1[j+i+1], v2[j+i+1] = calc_mf_f(m, thr_ct, N, is_clustering=False)   

    return v,v0,v1,v2 


def coefficient_f(thr_ct, smoothing, nside, itr, b, f):

    N = 12*nside*nside
    map_len = 2                    # sum of the number of lensing and clustering redshift bins
    array_len = map_len*thr_ct*3   # length of covariance array - multiply by 3 for 3 MFs

    # Fixed map run with iteration count
    v_all_fixed = np.zeros((itr,array_len)) 
    for i in range(itr):
        print("Simulating maps for iteration ", i)
        clustering_maps, lensing_maps = simulate_des_maps(0.3, 0.8, smoothing, nside, nmax=1)
        print("Computing functionals for iteration", i)
        v_fixed, v0_fixed, v1_fixed, v2_fixed = calc_mf_2maps_f(clustering_maps,lensing_maps,thr_ct,f)
        v_all_fixed[i] = np.concatenate((v0_fixed.flatten(),v1_fixed.flatten(),v2_fixed.flatten()))
        print(i)
    
    # stack iteration fixed variable versions of V0,V1,V2
    v_all_mean = np.zeros(array_len)
    for i in range(array_len):
        v_all_mean[i] = np.mean(v_all_fixed[:,i])
        
        
    # likelihood perpendicular line points (get constants from plotting notebook)
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = 0.8989639361571576*omega_m + 0.5303108191528527
    
    # calculate S_8
    S_8 = sigma_8 * (omega_m/0.3)**0.5329788249790618   # exponent value found in plotting notebook


    tic = time.perf_counter()
    
    # calculate MFs for each omega sigma pair along line
    V_all = np.zeros((b,array_len))
    c_map = np.zeros((b,len(clustering_maps),N)) 
    l_map = np.zeros((b,len(lensing_maps),N))

    for i in range(b):
        c_map[i], l_map[i] = simulate_des_maps(omega_m[i], sigma_8[i], smoothing, nside, nmax=1)
        v, v0, v1, v2 = calc_mf_2maps_f(c_map[i],l_map[i],thr_ct,f)
        V_all[i] = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        print(i)

    toc = time.perf_counter()
    print(round((toc - tic)/3600,2),'hrs')
    
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
            L[i] = -0.5 * (V_all[i] - v_all_mean) @ inv_cov @ (V_all[i] - v_all_mean)
    except:
        inv_cov2 = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov2)
        for i in range(b):
            d = (V_all[i] - v_all_mean)[good]
            L[i] = -0.5 * d @ inv_cov2 @ d
            
            
    coefficient = np.polyfit(S_8,L,2)
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    
    # output array
    c = np.array((thr_ct,smoothing,nside,constraining_power))
    
    
    ## save data ##

    # parent directory
    output_path = os.path.join(os.getcwd(), '2_Maps_Output')

    # save data in new subfolder
    parent_path = os.path.join(output_path, 'cov_fix')
    sub_path = os.path.join(parent_path, f't{thr_ct}_n{nside}_s{smoothing}_f{f}')
    

    try:
        os.mkdir(sub_path) 
    except:
        pass

    np.savetxt(os.path.join(sub_path, 'c.out'),c)
    np.savetxt(os.path.join(sub_path, 'V_all_fixed.out'),v_all_fixed)
    np.savetxt(os.path.join(sub_path, 'v_all_changing'),V_all)
    
    