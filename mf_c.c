#include "math.h"


void calculate_v12(int n, double * k, double * sq, double * frac, int nt, double vmin, double vspace, double * v1, double * v2){

    for (int b=0;b<nt; b++){
        v1[b] = 0;
        v2[b] = 0;
    }

    #pragma omp parallel
    {
        double v1_thread[nt];
        double v2_thread[nt];

        for (int b=0;b<nt; b++){
            v1_thread[b] = 0;
            v2_thread[b] = 0;
        }

        #pragma omp for nowait
        for (int i=0; i<n; i++){
            int b = floor((k[i] - vmin) / vspace);
            if ((b >= 0) && (b < nt)){
                v1_thread[b] += sq[i];
                v2_thread[b] += frac[i];
            }
        }

        #pragma omp critical 
        {
            for(int b=0; b<nt; b++){
                v1[b] += v1_thread[b];
                v2[b] += v2_thread[b];
            }
        }
    }  
}


