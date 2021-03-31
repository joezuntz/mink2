#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:11:14 2021

@author: ngrewal
"""

                                                                                                                                                            
import os
import numpy as np
import math
from mf import calc_mf_2maps
from cl import Cl_2maps
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import simulate_des_maps_bias

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]


# define an empty dictionary for mean and covariance of each nside and smoothing combo
dict_v = {}
dict_cov = {}


def likelihood_s(cosmo_params, smoothing=5, nside=256, thr_ct=10, sky_frac=1, m_type = 'c', save_L=False):
    
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
    sky_frac:   The percent of sky fraction in decimal format (i.e. 5% -> 0.05). The default is 1 (100%).
    m_type:     Clustering or lensing map.
    save_L:     Saves likelihood values in an output file if True.

    Returns
    -------
    L :         The likehood for the input parameter space.
    '''

    print('Cosmological parameter values: ',cosmo_params)

    # define cosmological parameters based on input array
    omega_b = 0.048
    omega_m = cosmo_params[0]
    h = 0.7
    n_s = 0.96
    sigma_8 = cosmo_params[1]
    b1 = 1.42
    b2 = 1.65
    b3 = 1.6
    b4 = 1.92
    b5 = 2

    if omega_m<0.2 or omega_m>0.4 or sigma_8<0.7 or sigma_8>0.9:
        return -math.inf
    
    '''    
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
    '''
    
    # calculate sky fraction
    frac = int(math.floor(sky_frac*12*nside**2))

    
    ## add try and except - check for value errors, and return -inf if needed
    
    try:
    
        # build new clustering and lensing maps
        cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside, nmax=1)
    
        # get analysis values from all iterations
        if (nside,smoothing,thr_ct,sky_frac) in dict_v:                       # find the corresponding workspace
            V = dict_v[nside,smoothing,thr_ct,sky_frac]
            cov = dict_cov[nside,smoothing,thr_ct,sky_frac]
        else:                                                                       # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
            V = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f1_Cl_{m_type}_1map.npy')    # this comes from '/disk01/ngrewal/Fiducial_Simulations'
            cov = np.cov(V[:frac].transpose())                                      # find the covariance    
            dict_v[nside,smoothing,thr_ct,sky_frac,] = V[:frac]                     # save the mean vector in the corresponding workspace
            dict_cov[nside,smoothing,thr_ct,sky_frac] = cov                         # save the covariance in the corresponding workspace                                                             
         
        # find analysis mean
        output_mean = np.mean(V[:frac],axis=0)                         # find the mean of the fiducial simulation MFs and Cls

        # power spectrum output for the first clustering map           
        if m_type=='c':
            output = Cl_2maps(cmaps,[],nside,frac).flatten()
       
        # power spectrum output for the first lensing map     
        if m_type=='l':
            output = Cl_2maps([],lmaps,nside,frac).flatten()

        
        # Find the inverse covariance
        i_cov = np.linalg.inv(cov)                           # find the inverse covariance  
        '''
        itr = len(V)                                          # find number of iterations
        N_ = itr-1                                            # number of iterations - 1
        p = len(V[0])                                         # number of data points (MFs, Cls, or both)
        i_cov = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov)      # find the inverse covariance with the Anderson-Hartlap correction
        '''
        # FIND LIKELIHOOD      
        diff = output - output_mean
        L = -0.5 * diff @ i_cov @ diff
        
        # return the likelihood
        print('Likelihood: ',L)
        
        # save likelihood if specified
        if save_L:
            prev_L = np.load('L.npy',allow_pickle=True)
            new_L = np.concatenate((prev_L,L),axis=None)
            print('Length of likelihood array: ',new_L.shape)
            np.save('L',new_L)

        return L#,output,output_mean
        
    except:
        raise
        #print('likelihood error')
        return -math.inf


