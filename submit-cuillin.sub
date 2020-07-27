#!/bin/bash -l
# number of processes to use, should prob be one for us here
#SBATCH -n 1

# amount of time (H:M:S) allocated to job - if not finished at time limit
# the job will die
#SBATCH -t 96:00:00

# name for the job - useful for monitoring later
#SBATCH -J mink

# number of CPUs to use 4 will usually make sense
#SBATCH --cpus-per-task=4

# memory allocated to job - again, if we go over this job will die
#SBATCH --mem=8G

cd $HOME/mink2

# these control how we use the different CPUs.
# leave alone! Unless change cpus-per-task above, in which
# case OMP_NUM_THREADS should match
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=4

# First time only - download the docker image
# Unlike docker, singularity puts this as a file in
# the current directory
if [ ! -f mink-latest.simg ]; then
    singularity pull docker://joezuntz/mink:latest
fi

# singularity is like docker - run in the same way
# except that the -V flag is not needed.  We run a shell
# script which in turn runs our python script
singularity run  ./mink-latest.simg ./run-cuillin.sh
