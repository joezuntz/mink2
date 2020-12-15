#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:36:00 2020

@author: ngrewal
"""
                                                                                                                                                            
import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from mf import *
from cl import *
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]


def likelihood(N,omega_b_min=0.048, omega_b_max=0.048,
                 omega_m_min=0.3,   omega_m_max=0.3,
                 h_min=0.7,         h_max=0.7,
                 n_s_min=0.96,      n_s_max=0.96,
                 sigma_8_min=0.8,   sigma_8_max=0.8,
                 b1_min=1.42,       b1_max=1.42,
                 b2_min=1.65,       b2_max=1.65,
                 b3_min=1.60,       b3_max=1.60,
                 b4_min=1.92,       b4_max=1.92,
                 b5_min=2.00,       b5_max=2.00,
                 smoothing=0, nside=1024, smoothing_r=20, nside_r=512, thr_ct=10, new_maps=True):
    
    # defining ranges for map simulation function
    omega_b_range = np.linspace(omega_b_min,omega_b_max,N)
    omega_m_range = np.linspace(omega_m_min,omega_m_max,N)
    h_range       = np.linspace(h_min,h_max,N)
    n_s_range     = np.linspace(n_s_min,n_s_max,N)
    sigma_8_range = np.linspace(sigma_8_min,sigma_8_max,N)
    b1_range      = np.linspace(b1_min,b1_max,N)
    b2_range      = np.linspace(b2_min,b2_max,N)
    b3_range      = np.linspace(b3_min,b3_max,N)
    b4_range      = np.linspace(b4_min,b4_max,N)
    b5_range      = np.linspace(b5_min,b5_max,N)
    
    # location of simulations (NOTE: this code only keeps current simulations in the folder - a new strategy is needed if maps need to be saved)
    map_path = '/disk01/ngrewal/Simulations'
    
    # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
    V = np.load('v_all.npy')                                       # this comes from '/disk01/ngrewal/Fiducial_Simulations'
    cov = np.cov(V.transpose())                                    # find the covariance                                                                                                                      
    i_cov = np.linalg.inv(cov)                                     # find the inverse covariance  
    vc_mean = np.mean(V,axis=0)                                    # find the mean of the fiducial simulation MFs and Cls                                                                                     
    L = np.zeros(len(N))                                           # empty array in which to populate likelihood values

    # load maps and do calculations N times
    for i in range(N):
        
        # build new maps if specified
        if new_maps==True:
            
            # remove existing clustering and lensing maps
            os.remove(os.path.join(map_path,f'cmaps_{i+1}.npy'))
            os.remove(os.path.join(map_path,f'lmaps_{i+1}.npy'))
            
            # build new clustering and lensing maps
            c_maps,l_maps = simulate_des_maps_bias(omega_b_range[i],
                                                   omega_m_range[i],
                                                   h_range[i],
                                                   n_s_range[i],
                                                   sigma_8_range[i],
                                                   b1_range[i], 
                                                   b2_range[i], 
                                                   b3_range[i], 
                                                   b4_range[i], 
                                                   b5_range[i], 
                                                   smoothing, nside)
            # save new clustering and lensing maps
            np.save(os.path.join(map_path, f'cmaps_{i+1}'),c_maps)  # clustering maps
            np.save(os.path.join(map_path, f'lmaps_{i+1}'),l_maps)  # lensing maps
            
        
        # load maps
        cmaps = np.load(os.path.join(map_path, f'cmaps_{i+1}.npy'))  # clustering maps                                                                                                                        
        lmaps = np.load(os.path.join(map_path, f'lmaps_{i+1}.npy'))  # lensing maps  
    
        ## simplify maps                                                                                                                                                                                    
        # reduce pixel size                                                                                                                                                                                 
        cmaps1 = hp.ud_grade(cmaps,nside_r)
        lmaps1 = hp.ud_grade(lmaps,nside_r)
        
        # smooth maps                                                                                                                                                                                       
        cmaps2 = np.zeros((len(cmaps),12*nside_r**2))
        lmaps2 = np.zeros((len(lmaps),12*nside_r**2)) 
        for i in range(len(cmaps)):
            cmaps2[i] = hp.smoothing(cmaps1[i],fwhm=smoothing_r)
        for i in range(len(lmaps)):
            lmaps2[i] = hp.smoothing(lmaps1[i],fwhm=smoothing_r)
                    
        # calculate MFs                                                                                                                                                                                     
        v,v0,v1,v2 = calc_mf_2maps(cmaps2,lmaps2,thr_ct)
        v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        
        # calculate Cls                                                                                                                                                                                     
        c = Cl_2maps(cmaps2,lmaps2,nside).flatten()
        
        # concatenate MFs and Cls
        vc = np.concatenate((v_all,c))
        
        # find the likelihood for a select amount of non-fiducial simulations                                                           
        diff = vc - vc_mean
        L[i] = -0.5 * diff @ i_cov @ diff
    
    np.save('LH',L)
    print(len(V))
        
