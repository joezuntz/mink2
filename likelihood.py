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





def likelihood(cosmo_params, smoothing=10, nside=512, thr_ct=10):
    
    # input needs to be an array not a dictionary
    
    '''
    Parameters
    ----------
    cosmo_params : [
        omega_b = Density of baryonic matter. The default is 0.048.
        omega_m = Density of all types of matter. The default is 0.3.
        h       = Rate of expansion of the Universe. The default is 0.7.
        n_s     = Spectral index. The default is 0.96.
        sigma_8 = Density perturbation amplitude. The default is 0.8.
        b1      = First galaxy bias bin. The default is 1.42.
        b2      = Second galaxy bias bin. The default is 1.65.
        b3      = Third galaxy bias bin. The default is 1.60.
        b4      = Fourth galaxy bias bin. The default is 1.92.
        b5      = Fifth galaxy bias bin. The default is 2.00.
        ]
    smoothing : Gaussian smoothing applied to maps in arcminutes. The default is 10.
    nside :     The number of pixels on each side of the map; pixels in the map total to 12*nside**2. The default is 512.
    thr_ct :    Number of map thresholds; corresponds to number of convergence maps calculations are done upon. The default is 10.

    Returns
    -------
    L :         The likehood for the input parameter space.
    '''

    # define cosmological parameters based on input array
    omega_b = cosmo_params[0]
    omega_m = cosmo_params[1]
    h = cosmo_params[2]
    n_s = cosmo_params[3]
    sigma_8 = cosmo_params[4]
    b1 = cosmo_params[5]
    b2 = cosmo_params[6]
    b3 = cosmo_params[7]
    b4 = cosmo_params[8]
    b5 = cosmo_params[9]
    
    # build new clustering and lensing maps
    cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside)

    # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
    V = np.load(f'vc_all_s{smoothing}_n{nside}.npy')                # this comes from '/disk01/ngrewal/Fiducial_Simulations'
    cov = np.cov(V.transpose())                                    # find the covariance                                                                                                                      
    i_cov = np.linalg.inv(cov)                                     # find the inverse covariance  
    vc_mean = np.mean(V,axis=0)                                    # find the mean of the fiducial simulation MFs and Cls                                                                                     
                         
    # calculate MFs                                                                                                                                                                                     
    v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct)
    v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
    
    # calculate Cls                                                                                                                                                                                     
    c = Cl_2maps(cmaps,lmaps,nside).flatten()
    
    # concatenate MFs and Cls
    vc = np.concatenate((v_all,c))
    
    # find the likelihood                                                     
    diff = vc - vc_mean
    L = -0.5 * diff @ i_cov @ diff
    
    # return the likelihood
    return L
        
