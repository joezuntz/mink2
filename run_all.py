#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:36:21 2021

@author: ngrewal
"""

import subprocess

# DRAFT OF SCRIPT THAT CREATES A SUITE OF SUBMISSION SCRIPTS

smoothing = sys.argv[1]
nside = sys.argv[2]
thr_ct = sys.argv[3]
sky_frac = sys.argv[4]
a_type = sys.argv[5]
m_type = sys.argv[6]

# make worker nodes a variable bc they change frequently
node_list = 'worker[001-036],worker[038-066],worker[075-076],worker[078-084]'

# RUN 3 ANALYSIS STEPS

# STEP 1: generate maps and measure observables
# STEP 2: combine observable iterations into a single array

# create a new script
calc_script = open(f'/home/ngrewal/mink2/sub/calc_{a_type}_{m_type}.sub',"w+")

calc_script.write(f'''
                  
#!/bin/bash                                                                                                                                       
#SBATCH --time=100:00:00                                                                                                                          
#SBATCH --array=1-4000                                                                                                                            
#SBATCH --cpus-per-task=8                                                                                                                         
#SBATCH --exclude={node_list}                                                                 
#SBATCH --nodes=1                                                                                                                                 
#SBATCH --tasks-per-node=1                                                                                                                        
#SBATCH --output='/disk01/ngrewal/logs/log.mf_cl_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}.%a.txt'                                                                                        

export LD_LIBRARY_PATH=${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=8
time srun -u -n1 python /home/ngrewal/mink2/calc_mf_cl.py {nside} {smoothing} {thr_ct} {sky_frac} {m_type}

# concatenate fiducial observables
python /home/ngrewal/mink2/fid_concat.py {a_type} {m_type}''')

calc_script.close()

# run script
s = subprocess.check_output(args = f'/home/ngrewal/mink2/sub/sbatch calc_{a_type}_{m_type}.sub')

# gets a central job id
calc_job_id = s.split()[-1]
print(calc_job_id)

# STEP 3: run an mcmc chain

# create a new script
mcmc_script = open(f'/home/ngrewal/mink2/sub/mcmc_{a_type}_{m_type}.sub',"w+")

mcmc_script.write(f'''
                                                                                              
#!/bin/bash                                                                                                                                       
#SBATCH --time=05:00:00                                                                                                                           
#SBATCH --cpus-per-task=32                                                                                                                        
#SBATCH --exclude={node_list}                                                                
#SBATCH --nodes=1                                                                                                                                 
#SBATCH --tasks-per-node=1                                                                                                                        
#SBATCH --output=log.mcmc_{a_type}_{m_type}.txt    
#SBATCH --dependency=afterok:{calc_job_id}                                                                                                       

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=32
#time mpiexec -n 1 python mcmc_simplified.py                                                                                                      
time python /home/ngrewal/mink2/mcmc.py {smoothing} {nside} {thr_ct} {sky_frac} {a_type} {m_type}''')

mcmc_script.close()


# run script
s = subprocess.check_output(args = f'/home/ngrewal/mink2/sub/sbatch mcmc_{a_type}_{m_type}.sub')





