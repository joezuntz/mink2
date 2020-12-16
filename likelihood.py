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


def likelihood(omega_b = 0.048, omega_m = 0.3, h = 0.7, n_s = 0.96, sigma_8 = 0.8, b1 = 1.42, b2 = 1.65, b3 = 1.60, b4 = 1.92, b5 = 2.00, smoothing=0, nside=1024, thr_ct=10):
    
    """
    Parameters
    ----------
    omega_b :   Density of baryonic matter. The default is 0.048.
    omega_m :   Density of all types of matter. The default is 0.3.
    h :         Rate of expansion of the Universe. The default is 0.7.
    n_s :       Spectral index. The default is 0.96.
    sigma_8 :   Density perturbation amplitude. The default is 0.8.
    b1 :        First galaxy bias bin. The default is 1.42.
    b2 :        Second galaxy bias bin. The default is 1.65.
    b3 :        Third galaxy bias bin. The default is 1.60.
    b4 :        Fourth galaxy bias bin. The default is 1.92.
    b5 :        Fifth galaxy bias bin. The default is 2.00.
    smoothing : Gaussian smoothing applied to maps in arcminutes. The default is 0.
    nside :     The number of pixels on each side of the map; pixels in the map total to 12*nside**2. The default is 1024.
    thr_ct :    Number of map thresholds; corresponds to number of convergence maps calculations are done upon. The default is 10.

    Returns
    -------
    L :         The likehood for the input parameter space.

    """
    

    # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
    V = np.load('vc_all_s{smoothing}_n{nside}.npy')                # this comes from '/disk01/ngrewal/Fiducial_Simulations'
    cov = np.cov(V.transpose())                                    # find the covariance                                                                                                                      
    i_cov = np.linalg.inv(cov)                                     # find the inverse covariance  
    vc_mean = np.mean(V,axis=0)                                    # find the mean of the fiducial simulation MFs and Cls                                                                                     
     
    # build new clustering and lensing maps
    cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside)
                                                                                                                                                                             
    ## simplify maps ##

    # reduce pixel size                                                                                                                                                                                 
    cmaps1 = hp.ud_grade(cmaps,nside)
    lmaps1 = hp.ud_grade(lmaps,nside)
    
    # smooth maps   
    smoothing_r = np.radians(smoothing_a/60)                # convert arcmin to degrees                                                                                                                                                                                    
    cmaps2 = np.zeros((len(cmaps),12*nside**2))
    lmaps2 = np.zeros((len(lmaps),12*nside**2)) 
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
    
    # find the likelihood                                                     
    diff = vc - vc_mean
    L = -0.5 * diff @ i_cov @ diff
    
    # return the likelihood
    return L
        
