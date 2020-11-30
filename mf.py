import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary("./mf_c.so")
calculate_v12_c = lib.calculate_v12


# int n, double * v, double * sq, double * frac, int nt, double vmin, double vspace, double * v1, double * v2


c_double_p = ctypes.POINTER(ctypes.c_double)


calculate_v12_c.argtypes = [
    ctypes.c_int, # n
    c_double_p,   # v
    c_double_p,   # sq
    c_double_p,   # frac
    ctypes.c_int, # nt
    ctypes.c_double, # vmin
    ctypes.c_double, # vspace
    c_double_p, # v1
    c_double_p, # v2 
]

calculate_v12_c.restype = None


def V_12(v,k,kx,ky,kxx,kxy,kyy):
    
    vmin = v.min()                      # threshold min
    vmax = v.max()                      # threshold max
    vspace = (vmax-vmin)/len(v)         # threshold array bin size

    N = k.size
    nt = v.size

    output1 = np.zeros(len(v))
    output2 = np.zeros(len(v))
  
    # define MF functions
    sq = np.sqrt(kx**2 + ky**2)
    frac = (2*kx*ky*kxy - (kx**2)*kyy - (ky**2)*kxx)/(kx**2 + ky**2)
    

    calculate_v12_c(N, 
                    k.ctypes.data_as(c_double_p),
                    sq.ctypes.data_as(c_double_p),
                    frac.ctypes.data_as(c_double_p),
                    nt,
                    vmin,
                    vspace,
                    output1.ctypes.data_as(c_double_p),
                    output2.ctypes.data_as(c_double_p),
    )


    
    output1 = output1 / (4*N)
    output2 = output2 / (2*np.pi*N)    
    return output1,output2

