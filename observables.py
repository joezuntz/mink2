#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:47:48 2021

@author: ngrewal
"""
import numpy as np
import math
import os
from mf import calc_mf_2maps
from cl import Cl_2maps
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import simulate_des_maps_bias

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]



def observables(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside, thr_ct, sky_frac, a_type, m_type, seed=10291995):

    # build new clustering and lensing maps
    cmaps,lmaps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside, seed)
    
    # calculate sky fraction
    frac = int(math.floor(sky_frac*12*nside**2))
    
    # power spectrum (Cl) analysis only
    if a_type=='Cl':

        # clustering and lensing maps  
        if m_type=='c+l':                                                                                                                                                                                  
            output = Cl_2maps(cmaps,lmaps,nside,frac).flatten()
            
        # clustering only
        elif m_type=='c':
            output = Cl_2maps(cmaps,[],nside,frac).flatten()
            
        # lensing only
        elif m_type=='l':
            output = Cl_2maps([],lmaps,nside,frac).flatten()
    
        
    # Minkowski functional analysis only
    if a_type=='MF':
            
        # clustering and lensing maps
        if m_type=='c+l':                                                                                                                                                                                   
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)
            
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)
                
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac) 
                 
        output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
    
    
    # Minkowski functional and power spectrum analysis
    if a_type=='MF+Cl':
                     
        # clustering and lensing maps  
        if m_type=='c+l':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)     # calculate MFs
            c = Cl_2maps(cmaps,lmaps,nside,frac)                    # calculate Cls
           
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)        # calculate MFs
            c = Cl_2maps(cmaps,[],nside,frac)                           # calculate Cls
                
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)        # calculate MFs
            c = Cl_2maps([],lmaps,nside,frac)                           # calculate Cls
            
        # concatenate MFs and Cls
        output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))

                
    return output
        
        
        
'''another try - less code, but takes more time
        
        # TYPES OF MAPS 
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
    
        # TYPES OF ANALYSIS 
        # Minkowski functional and power spectrum analysis
        if a_type=='MF+Cl':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))
        
        # Minkowski functional analysis only
        if a_type=='MF':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        
        # power spectrum (Cl) analysis only
        if a_type=='Cl':
            output = c.flatten()
        '''
