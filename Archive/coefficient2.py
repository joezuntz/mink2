#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:17:24 2020

@author: nishagrewal
"""

#import pyccl
import healpy as hp
import numpy as np
#import matplotlib.pyplot as plt
import math
import os
import time

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

# calculate pixel derivatives

def map_derivatives(m):
    """
    Compute the derivatives of a map
    """
    npix = m.size
    nside = hp.npix2nside(npix)
    
    # Take the spherical Fourier transform of the map
    alm = hp.map2alm(m)

    # Convert back to real space and also get the first derivatives
    m_map, der_theta_map, der_phi_map = hp.alm2map_der1(alm, nside)

    # Fourier transform the first derivatives
    der_theta_alm = hp.map2alm(der_theta_map)
    der_phi_alm = hp.map2alm(der_phi_map)

    # Convert the first derivatives back to real space and also get the second derivatives
    der_theta_map2, der2_theta2_map, der2_theta_phi_map = hp.alm2map_der1(der_theta_alm, nside)
    der_phi_map2, der2_phi_theta_map, der2_phi2_map = hp.alm2map_der1(der_phi_alm, nside)

    # return all
    return [der_theta_map, der_phi_map, der2_theta2_map, der2_phi2_map, der2_theta_phi_map]


### MF functions

def V_0(v,k,N):

    output = []
    for i in v:

        # find the counts of pixels where the pixel height is greater than that threshold value
        count = (k > i).sum()
        output = np.append(output,count)
        
    output = output/N    
    return output


def V_12(v,k,kx,ky,kxx,kxy,kyy,N):
    
    vmin = v.min()                      # threshold min
    vmax = v.max()                      # threshold max
    vspace = (vmax-vmin)/len(v)         # threshold array bin size

    output1 = np.zeros(len(v))
    output2 = np.zeros(len(v))
  
    # define MF functions
    sq = np.sqrt(kx**2 + ky**2)
    frac = (2*kx*ky*kxy - (kx**2)*kyy - (ky**2)*kxx)/(kx**2 + ky**2)
    
    indices = np.floor((k-vmin)/vspace)
    
    # find the closest threshold value for every pixel    
    for i,pixel in enumerate(k):
        index = int(indices[i])
        
        # filter out values outside valid indeces        
        if 0 <= index < len(v): 
            output1[index] += sq[i]
            output2[index] += frac[i] 
    
    output1 = output1/(4*N)
    output2 = output2/(2*np.pi*N)    
    return output1,output2

## CALC MFs
# calculate MFs for a single map
    
def calc_mf(m,thr_ct,N,is_clustering):

    v_0 = np.zeros(thr_ct)
    v_1 = np.zeros(thr_ct)
    v_2 = np.zeros(thr_ct)

    dm_dtheta, dm_dphi, d2m_dtheta2, d2m_dphi2, d2m_dtheta_dphi = map_derivatives(m)

    std_dev = np.std(m) 
    
    if is_clustering is True:
        v = np.linspace(0,6*std_dev,thr_ct)              # clustering map range
    else:
        v = np.linspace(-3*std_dev, 3*std_dev,thr_ct)    # lensing map range

    v_0 = V_0(v, m, N)
    v_1,v_2 = V_12(v, m, dm_dphi, dm_dtheta, d2m_dphi2, d2m_dtheta_dphi, d2m_dtheta2, N)
    
    return v,v_0,v_1,v_2


# calculate MFs for both types of maps with different galaxy biases
    
def calc_mf_2maps2(clustering_maps,lensing_maps,thr_ct,N):

    map_len=2

    # find MFs for fixed parameter simulation   
    v = np.zeros((map_len,thr_ct))
    v0 = np.zeros((map_len,thr_ct))    
    v1 = np.zeros((map_len,thr_ct))
    v2 = np.zeros((map_len,thr_ct))

    v[0], v0[0], v1[0], v2[0] = calc_mf(clustering_maps[0], thr_ct, N, is_clustering=True)
    v[1], v0[1], v1[1], v2[1] = calc_mf(lensing_maps[0], thr_ct, N, is_clustering=False)  

    return v,v0,v1,v2 


# calculate constraining power

def coefficient2(thr_ct, smoothing, nside, itr, b):

    N = 12*nside*nside             # total number of pixels
    map_len = 2                    # sum of the number of lensing and clustering redshift bins
    array_len = map_len*thr_ct*3   # length of covariance array - multiply by 3 for 3 MFs


    # Fixed map run with iteration count
    v_all_fixed = np.zeros((itr,array_len)) 
    for i in range(itr):
        print("Simulating maps for iteration ", i)
        clustering_maps, lensing_maps = simulate_des_maps(0.3, 0.8, smoothing, nside)
        print("Computing functionals for iteration", i)
        v_fixed, v0_fixed, v1_fixed, v2_fixed = calc_mf_2maps2(clustering_maps,lensing_maps,thr_ct,N)
        v_all_fixed[i] = np.concatenate((v0_fixed.flatten(),v1_fixed.flatten(),v2_fixed.flatten()))
        print(i)

    # covariance
    cov = np.cov((v_all_fixed.transpose()))
    
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
        c_map[i], l_map[i] = simulate_des_maps(omega_m[i], sigma_8[i], smoothing, nside)
        v, v0, v1, v2 = calc_mf_2maps2(c_map[i],l_map[i],thr_ct,N)
        V_all[i] = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        print(i)

    toc = time.perf_counter()
    print(round((toc - tic)/3600,2),'hrs')

    # singular covariance matrix workaround
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
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    
    # output array
    c = np.array((thr_ct,smoothing,nside,constraining_power))
    
    
    ## save data ##

    # parent directory
    output_path = os.path.join(os.getcwd(), '2_Maps_Output')

    # save data in new subfolder
    parent_path = os.path.join(output_path, 'Simplified')
    sub_path = os.path.join(parent_path, f't{thr_ct}_n{nside}_s{smoothing}')
    

    try:
        os.mkdir(sub_path) 
    except:
        pass

    np.savetxt(os.path.join(sub_path, 'c.out'),c)
    np.savetxt(os.path.join(sub_path, 'V_all_fixed.out'),v_all_fixed)
    np.savetxt(os.path.join(sub_path, 'v_all_changing'),V_all)
    
    
    
    
    
    
