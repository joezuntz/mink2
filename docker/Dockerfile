FROM continuumio/miniconda3:4.8.2

# Install conda compiler and other background packages
RUN conda install -y gcc_linux-64 gxx_linux-64 gfortran_linux-64 fftw astropy scipy  \
    && conda clean -a


RUN conda install -y  -c conda-forge jupyterlab pyccl healpy emcee && conda clean -a

# Make symlinks to the compilers
RUN ln -s `which x86_64-conda_cos6-linux-gnu-gcc` /usr/local/bin/gcc
RUN ln -s `which x86_64-conda_cos6-linux-gnu-g++` /usr/local/bin/g++
RUN ln -s `which x86_64-conda_cos6-linux-gnu-gfortran` /usr/local/bin/gfortran
RUN ln -s `which x86_64-conda_cos6-linux-gnu-ar` /usr/local/bin/ar
RUN ln -s `which x86_64-conda_cos6-linux-gnu-ld` /usr/local/bin/ld
RUN ln -s `which x86_64-conda_cos6-linux-gnu-ranlib` /usr/local/bin/ranlib


RUN conda install -y -c conda-forge gsl make astropy healpix_cxx \
    && conda clean -a
RUN apt-get install -y healpy-data

RUN cd /opt \
    && git clone --branch 1.2 https://github.com/joezuntz/flask \
    && cd flask/src \
    && make \
    && cp ../bin/* /usr/local/bin \
    && rm -r /opt/flask

# RUN cd /opt \
#     && wget https://github.com/hsxavier/flask/archive/master.zip \
#     && git clone https://github.com/hsxavier/flask \
#     && cd flask \

# #  From the NERSC docs
# ARG mpich=3.3
# ARG mpich_prefix=mpich-$mpich

# RUN conda install -y make \
#     && conda clean -a


# RUN /sbin/ldconfig


# Prevents a crash wheere matplotlib looks for a QT backend.
ENV MPLBACKEND agg


# FROM ubuntu:18.04
# MAINTAINER joezuntz@googlemail.com
# #Joe's note to himself.  Compile this with: docker build -t joezuntz/cosmosis-base .
# #then docker push joezuntz/cosmosis-base

# # Basic compilers and tools dependencies
# RUN apt-get update -y && apt-get install -y gcc g++ gfortran \
#     cmake swig wget make libopenblas-dev libchealpix-dev \
#     pkg-config curl python3.7    python3-distutils python3.7-dev python3-pip \
#     libcfitsio-dev  libfftw3-dev   git  libgsl-dev autoconf \
#     && apt-get clean all

# RUN apt-get update -y && apt-get install -y libbz2-dev \
#     && apt-get clean all

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
# RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10



# ENV LD_LIBRARY_PATH /usr/local/lib
# ENV C_INCLUDE_PATH /usr/local/include

# # Ceci

# RUN pip install --no-cache-dir --upgrade pip

# # fitsio needs this to be installed first
# RUN pip install --no-cache-dir numpy cffi

# # Basic python requirements
# RUN pip install --no-cache-dir \
#                 Cython \
#                 nose \
#                 pyparsing \
#                 pyyaml \
#                 scikit-learn \
#                 setuptools \
#                 astropy \
#                 scipy \
#                 fitsio \
#                 healpy \
#                 six \
#                 threadpoolctl \
#                 emcee \
# #                 pyccl \
# #                 camb \
# #                 ipython

# RUN pip install --no-cache-dir jupyter

# # These prevent errors all over the place on NERSC
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# ENV INSIDE_DOCKER 1
WORKDIR /root/mink
CMD ["/bin/bash", "-l"]
