#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:36:00 2020

@author: ngrewal
"""

import numpy as np
import math

from observables import observables


# define an empty dictionary for mean and covariance of each nside and smoothing combo
dict_v = {}
dict_cov = {}


def likelihood(cosmo_params, smoothing=10, nside=512, thr_ct=10, sky_frac=1, a_type='MF', m_type='l', save_L = False, return_all=False):
    
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
    a_type:     Analysis type. Minkowski functional, power spectrum (Cl), or both. The default is both (MF+Cl).
    m_type:     Map type. Clustering, lensing, or both. The default is both ('c+l').
    return_all: Returns likelihood, mean array, covariance, analysis mean array, Minkowski thresholds, MF_0, MF_1, MF_2, Cls
    save_L:     Saves likelihood values in an output file if True.

    Returns
    -------
    L :         The likehood for the input parameter space.
    '''

    print('Cosmological parameter values: ',cosmo_params)

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


    # cosmological parameter priors
    if any(x<0 for x in cosmo_params) or omega_b<0.047 or omega_b>0.049 or omega_m<0.1 or omega_m>0.6 or h<0.5 or h>0.9 or n_s<0.9 or n_s>1.1 or sigma_8<0.3 or sigma_8>1.2:
        print('Likelihood: inf')
        return -math.inf
    
    ## add try and except - check for value errors, and return -inf if needed
    
    try:
        
        # get mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
        if (nside,smoothing,thr_ct,sky_frac,a_type,m_type) in dict_v:                       # find the corresponding workspace if it exists
            V = dict_v[nside,smoothing,thr_ct,sky_frac,a_type,m_type]
            cov = dict_cov[nside,smoothing,thr_ct,sky_frac,a_type,m_type]
        else:                                                                                     
            V = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}.npy')  # this comes from '/disk01/ngrewal/Fiducial_Simulations'
            cov = np.cov(V.transpose())                                                            # find the covariance    
            dict_v[nside,smoothing,thr_ct,sky_frac,a_type,m_type] = V                              # save the mean vector in the corresponding workspace
            dict_cov[nside,smoothing,thr_ct,sky_frac,a_type,m_type] = cov                          # save the covariance in the corresponding workspace                                                             
        fiducial_mean = np.mean(V,axis=0)                         # find the mean of the fiducial simulation MFs and Cls
         
        # Find the inverse covariance
        #i_cov = np.linalg.inv(cov)                           # find the inverse covariance  
        itr = len(V)                                          # find number of iterations
        N_ = itr-1                                            # number of iterations - 1
        p = len(V[0])                                         # number of data points (MFs, Cls, or both)
        i_cov = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov)      # find the inverse covariance with the Anderson-Hartlap correction

        # get MF and/or Cl observables given input parameters
        output = observables(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside, thr_ct, sky_frac, a_type, m_type, seed = 10291995)

        # FIND LIKELIHOOD      
        diff = output - fiducial_mean
        L = -0.5 * diff @ i_cov @ diff
        
        # return the likelihood
        print('Likelihood: ',L)
        
        # save likelihood if specified
        if save_L:
            print(L.shape)
            prev_L = np.load('L.npy')
            print('loaded')
            new_L = np.append((prev_L,L),axis=None)
            print('Length of likelihood array: ',new_L.shape)
            np.save('L',new_L)
        
        if return_all:
            return L,V,cov,fiducial_mean,output
        else:
            return L
        
    except:
        print('Likelihood: -inf')
        raise
        return -math.inf


