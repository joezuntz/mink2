# set up the conda install.  In docker this happens
# automatically but not in singularity
source /opt/conda/bin/activate

# run our script - can change this to launch anything
python run-calculation.py
