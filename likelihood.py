#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:36:00 2020

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


def likelihood(cosmo_params, smoothing=10, nside=512, thr_ct=10, sky_frac=1, a_type='MF+Cl', m_type='c+l', return_all=False):
    
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
    
    # calculate sky fraction
    frac = int(math.floor(sky_frac*12*nside**2))

    
    ## add try and except - check for value errors, and return -inf if needed
    
    try:
    
        # build new clustering and lensing maps
        cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside)
    
        if (nside,smoothing,thr_ct,sky_frac,a_type,m_type) in dict_v:                       # find the corresponding workspace
            V = dict_v[nside,smoothing,thr_ct,sky_frac,a_type,m_type]
            cov = dict_cov[nside,smoothing,thr_ct,sky_frac,a_type,m_type]
        else:                                                                                      # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
            V = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f1_{a_type}_{m_type}.npy')           # this comes from '/disk01/ngrewal/Fiducial_Simulations'
            cov = np.cov(V[:frac].transpose())                                                     # find the covariance    
            dict_v[nside,smoothing,thr_ct,sky_frac,a_type,m_type] = V[:frac]                       # save the mean vector in the corresponding workspace
            dict_cov[nside,smoothing,thr_ct,sky_frac,a_type,m_type] = cov                          # save the covariance in the corresponding workspace                                                             
         
        i_cov = np.linalg.inv(cov)                              # find the inverse covariance  
        output_mean = np.mean(V[:frac],axis=0)                         # find the mean of the fiducial simulation MFs and Cls
           
        
        ## BLOCK COMMENT ##
        # # Minkowski functional and power spectrum analysis
        # if a_type=='MF+Cl':
                      
        #     # clustering and lensing maps  
        #     if m_type=='c+l':
        #         v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)     # calculate MFs
        #         c = Cl_2maps(cmaps,lmaps,nside,frac)                    # calculate Cls
            
        #     # clustering only
        #     elif m_type=='c':
        #         v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)        # calculate MFs
        #         c = Cl_2maps(cmaps,[],nside,frac)                           # calculate Cls
                
        #     # lensing only
        #     elif m_type=='l':
        #         v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)        # calculate MFs
        #         c = Cl_2maps([],lmaps,nside,frac)                           # calculate Cls
            
        #     # concatenate MFs and Cls
        #     #output = np.concatenate((np.concatenate((v0.flatten(),v1.flatten(),v2.flatten())),c.flatten()))
        #     output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))
        
        # # Minkowski functional analysis only
        # if a_type=='MF':
            
        #     # clustering and lensing maps
        #     if m_type=='c+l':                                                                                                                                                                                   
        #         v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)
            
        #     # clustering only
        #     elif m_type=='c':
        #         v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)
                
        #     # lensing only
        #     elif m_type=='l':
        #         v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)
            
        #     output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
         
        # # power spectrum (Cl) analysis only
        # if a_type=='Cl':

        #     # clustering and lensing maps  
        #     if m_type=='c+l':                                                                                                                                                                                  
        #         output = Cl_2maps(cmaps,lmaps,nside,frac).flatten()
            
        #     # clustering only
        #     elif m_type=='c':
        #         output = Cl_2maps(cmaps,[],nside,frac).flatten()
            
        #     # lensing only
        #     elif m_type=='l':
        #         output = Cl_2maps([],lmaps,nside,frac).flatten()
        
        
        
        '''another try - less code, but may take more time'''
        
        ''' TYPES OF MAPS '''
        # clustering and lensing maps  
        if m_type=='c+l':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)     # calculate MFs
            c = Cl_2maps(cmaps,lmaps,nside,frac)                    # calculate Cls
        
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)        # calculate MFs
            c = Cl_2maps(cmaps,[],nside,frac)                       # calculate Cls
            
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)        # calculate MFs
            c = Cl_2maps([],lmaps,nside,frac)                       # calculate Cls
    
        ''' TYPES OF ANALYSIS '''
        # Minkowski functional and power spectrum analysis
        if a_type=='MF+Cl':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))
        
        # Minkowski functional analysis only
        if a_type=='MF':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        
        # power spectrum (Cl) analysis only
        if a_type=='Cl':
            output = c.flatten()
        
       
        ''' FIND LIKELIHOOD '''
        diff = output - output_mean
        L = -0.5 * diff @ i_cov @ diff
        
        # return the likelihood
        print(L)
        
        if return_all:
            return L,V[:frac],cov,output_mean,v,v0,v1,v2,c
        else:
            return L
        
    except:
        raise
        #print('likelihood error')
        return -math.inf





    ''' code capable of MF + Cl analysis only with both map types
    try:
    
        # build new clustering and lensing maps
        cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside)
    
        if (nside,smoothing) in dict_v:                  # find the corresponding workspace
            V = dict_v[nside,smoothing]
            cov = dict_cov[nside,smoothing]
        else:                                                  # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
            V = np.load(f'vc_all_s{smoothing}_n{nside}.npy')   # this comes from '/disk01/ngrewal/Fiducial_Simulations'
            cov = np.cov(V.transpose())                        # find the covariance    
            dict_v[nside,smoothing] = V                  # save the mean vector in the corresponding workspace
            dict_cov[nside,smoothing] = cov              # save the covariance in the corresponding workspace                                                             
        
        i_cov = np.linalg.inv(cov)                             # find the inverse covariance  
        vc_mean = np.mean(V,axis=0)                            # find the mean of the fiducial simulation MFs and Cls
                                 
        # calculate MFs                                                                                                                                                                                     
        v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)
        v_all = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        
        # calculate Cls                                                                                                                                                                                     
        c = Cl_2maps(cmaps,lmaps,nside,frac).flatten()
        
        # concatenate MFs and Cls
        vc = np.concatenate((v_all,c))
        
        # find the likelihood                                                     
        diff = vc - vc_mean
        L = -0.5 * diff @ i_cov @ diff
        
        # return the likelihood
        #print('ok')
        
        if return_all:
            return L,v_all,c,vc,vc_mean,cov
        else:
            return L
        
    except:
        raise
        #print('likelihood error')
        return -math.inf
    '''


    ''' non dictionary, non try code
    # build new clustering and lensing maps
    cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside)
    
    # load mean of fiducial simulation MF + Cl arrays (Note: assumes mean has been calculated already)
    V = np.load(f'vc_all_s{smoothing}_n{nside}.npy')   # this comes from '/disk01/ngrewal/Fiducial_Simulations'
    cov = np.cov(V.transpose())                        # find the covariance    
    i_cov = np.linalg.inv(cov)                         # find the inverse covariance  
    vc_mean = np.mean(V,axis=0)                        # find the mean of the fiducial simulation MFs and Cls                                                                                     
                             
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

    '''
        
