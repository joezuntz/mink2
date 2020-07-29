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
    
    
    
    
    
    
'''
    
def coefficient_old(thr_ct,smoothing, nside):
    
    tic = time.perf_counter()
    
    N = 12*nside*nside   

    clustering_maps, lensing_maps = simulate_des_maps(0.3, 0.8, smoothing, nside)
    toc1 = time.perf_counter()
    print('Fixed parameter simulated map created',round(toc1-tic,2)/60)


    v_fixed, v0_fixed, v1_fixed, v2_fixed = calc_mf_2maps(clustering_maps,lensing_maps,thr_ct,N)
    toc2 = time.perf_counter()    
    print('MFs calculated for fixed map',round(toc2-toc1,2)/60)    


    # mean per threshold value
    mean_v0 = np.zeros(thr_ct)
    mean_v1 = np.zeros(thr_ct)
    mean_v2 = np.zeros(thr_ct)
    
    for i in range(0,thr_ct):
        mean_v0[i] = np.mean(v0_fixed[:,i])
        mean_v1[i] = np.mean(v1_fixed[:,i])
        mean_v2[i] = np.mean(v2_fixed[:,i])
        
    # used in likelihood calculation
    mean_array = np.append(np.append(mean_v0,mean_v1),mean_v2)
    toc3 = time.perf_counter()       
    print('Calculated mean for fixed map',round(toc3-toc2,2)/60)
   
    
    # stack 100 iteration fixed variable versions of V0,V1,V2 to find covariance
    v_all_mean = np.concatenate((v0_fixed,v1_fixed,v2_fixed),axis=1)
    
    cov = np.cov((v_all_mean.transpose()))
    toc4 = time.perf_counter()      
    print('Calculated covariance for fixed map',round(toc4-toc3,2)/60)
    
    map_len = len(clustering_maps)+len(lensing_maps)



    # generate line perpendicular to likelihood
    b=10
    perpendicular_slope = 0.8989639361571576
    y_intercept = 0.5303108191528527
    
    omega_m = np.linspace(0.2,0.4,b)
    sigma_8 = perpendicular_slope*omega_m + y_intercept

    toc5 = time.perf_counter()  
    print('Perpendicular line created',round(toc5-toc4,2)/60)



    ## run MFs along perpendicular line
    
    v = np.zeros((b,map_len,thr_ct))
    v0 = np.zeros((b,map_len,thr_ct))
    v1 = np.zeros((b,map_len,thr_ct))
    v2 = np.zeros((b,map_len,thr_ct))
    
    # if something breaks, check these dimensions (~ 90% sure theyre accurate)
    c_map = np.zeros((b,len(clustering_maps),N)) 
    l_map = np.zeros((b,len(lensing_maps),N))
    
    for i in range(b):
        c_map[i],l_map[i] = simulate_des_maps(omega_m[i], sigma_8[i], smoothing, nside)
        v[i],v0[i],v1[i],v2[i] = calc_mf_2maps(c_map[i],l_map[i],thr_ct,N)
        print(i)
        
    toc6 = time.perf_counter()
    print('It took',round((toc6 - toc5)/3600,2),'hrs to run the simulation and calculate the MFs for points along the perpendicular line')
    
    # reformat maps into 2D to be able to save them
    
    C_map = c_map.reshape((b*len(clustering_maps),N))
    L_map = l_map.reshape((b*len(lensing_maps),N))
    
    # combine arrays to save and use v_all in likelihood calculation
    v = np.concatenate(v)
    v_all = np.concatenate((np.concatenate(v0),np.concatenate(v1),np.concatenate(v2)),axis=1)
    
    
    
    # calculate S_8
    S_8 = np.zeros(b)
    
    for i in range(b):
        S_8[i] = sigma_8[i] * (omega_m[i]/0.3)**0.5329788249790618  # exponent value found in plotting notebook
    
    toc7 = time.perf_counter()  
    print('S_8 calculated',round(toc7-toc6,2)/60)
    
    
    # sometimes the determinant of the covariance matrix is zero 
    #    -> singular matrix -> cannot find the inverse
    # singular matrix workaround I found online: add a small amount of noise to the cov matrix
    cov_w_noise = cov
    
    for i in range(30):
        for j in range(30):
            cov_w_noise[i][j] = cov[i][j] + np.random.random()*0.0001
            
    
    # calculate the likelihood          
    L = np.zeros(b)

    try:
        for i in range(b):
            L[i] = -0.5 * (v_all[i] - mean_array) @ np.linalg.inv(cov) @ (v_all[i] - mean_array)
    
    except:
        for i in range(b):
            L[i] = -0.5 * (v_all[i] - mean_array) @ np.linalg.inv(cov_w_noise) @ (v_all[i] - mean_array)
        

    toc9 = time.perf_counter()          
    print('Likelihood calculated',round(toc9-toc7,2)/60)
    
    
    
    # calculate the coefficient
    coefficient = np.polyfit(S_8,L,2)
    constraining_power = np.sqrt(-1 / (2*coefficient[0]))
    
    # save with 2 values bc python wont save an array with one value
    c = np.array((constraining_power,constraining_power))
    
    
    
    ## SAVE DATA ##
    
    # save data by threshold range, nside, and smoothing

    output_path = os.path.join(os.getcwd(), '2_Maps_Output')
    
    # parent folder for input variable combination
    path = os.path.join(output_path, f't{thr_ct}_n{nside}_s{smoothing}')
    
    try:
        os.mkdir(path) 
    except:
        pass
    
    # save coefficient    
    np.savetxt(os.path.join(path, 'c.out'),c)
    
    # save simulated maps 
    np.savetxt(os.path.join(path, 'c_map.out'),C_map)
    np.savetxt(os.path.join(path, 'l_map.out'),L_map)
    
    # save fixed parameter data that is used to find the mean and covariance
    fixed_path = os.path.join(path, 'fixed')
    
    try:
        os.mkdir(fixed_path)
    except:
        pass
    
    np.savetxt(os.path.join(fixed_path, 'v.out'),v_fixed)
    np.savetxt(os.path.join(fixed_path, 'v0.out'),v0_fixed)
    np.savetxt(os.path.join(fixed_path, 'v1.out'),v1_fixed)
    np.savetxt(os.path.join(fixed_path, 'v2.out'),v2_fixed)
    
    
    # save changing parameter data that is used to find the likelihood
    changing_path = os.path.join(path, 'changing')
    
    try:
        os.mkdir(changing_path)
    except:
        pass
    
    np.savetxt(os.path.join(changing_path, 'v.out'),v)
    np.savetxt(os.path.join(changing_path, 'v_all.out'),v_all)

    
    toc10 = time.perf_counter()          
    print('Maps and MFs saved',round(toc10-toc9,2)/60)
    
    return constraining_power


'''
