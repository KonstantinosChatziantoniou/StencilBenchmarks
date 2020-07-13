#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include "../headers/my_kernel_regs_4.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define TRUE 1
#define FALSE 0
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}




__global__ void my_32_4(float* g_input, float* g_output, // float* g_vsg, = 1
                        const int dimx, const int dimy, const int dimz)
{
    #define BDIMX 32 // tile (and threadblock) size in x
    #define BDIMY 32 // tile (and threadblock) size in y
    #define radius 4 // half of the order in space (k/2)

    __shared__ float tile[BDIMY+2*radius][BDIMX + 2*radius];
    //__shared__ float s_out[BDIMY][BDIMX][2*radius + 1];
    //float* z_data = all_z_data[threadIdx.x + threadIdx.y*BDIMX];
    int ix = blockIdx.x*blockDim.x + threadIdx.x  + radius;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + radius;
    int stride = dimx*dimy; 
    int in_idx = iy*dimx + ix;                  // index for reading input
    //0int out_idx = in_idx + (radius-1)*stride;   // index for writing output

    int len = 2*radius + BDIMX; // Assuming BDIMX = BDIMY

    // int txr = threadIdx.x + radius;              
    // int tyr = threadIdx.y + radius;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int txr = threadIdx.x + radius;              // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    int tyr = threadIdx.y + radius;
    int tidx = threadIdx.x;//(threadIdx.x + 16*threadIdx.y)%32;
    int tidy = threadIdx.y;//(threadIdx.x + 16*threadIdx.y)/32;
    int tidix = blockIdx.x*blockDim.x + tidx;
    int tidiy = blockIdx.y*blockDim.y + tidy;
    //int stride = dimx*dimy; 
    int tid_in_idx = tidiy*dimx + tidix;        
    // Init out shared array
    // for(int i = 0; i < (2*radius+1); i++){
    //     s_out[ty][tx][i] = 0;   
    // }
    float behind4 = 0;
    float behind3 = 0;
    float behind2 = 0;
    float behind1 = 0;
    float current = 0;
    float infront1 = 0;
    float infront2 = 0;
    float infront3 = 0;
    float infront4 = 0;
    tid_in_idx += (radius)*stride;
    for(int z = radius; z < dimz - radius; z++){
        // read tile with halo
        __syncthreads();
        for(int i = 0; i < len; i += 32){
            for(int j = 0; j < len; j += 32){
                if(tidx + i < len && tidy + j < len){
                    tile[j+tidy][i+tidx] = g_input[tid_in_idx + i + j*dimx];
                    //printf("%d %d %d %d\n",tx,ty, j+tidy, i+tidx);
                }
            }
        }

        __syncthreads();
        current += 3*tile[tyr][txr];
        // for(int i = 1; i <= radius; i++){
        //     current += (tile[tyr][txr+i]
        //     + tile[tyr+i][txr] + tile[tyr-i][txr]
        //     + tile[tyr][txr-i]);
        //     // s_out[ty][tx][4+i] += tile[tyr][txr];
        //     // s_out[ty][tx][i-1] += tile[tyr][txr];
        // }
        current += (tile[tyr][txr+1]
            + tile[tyr+1][txr] + tile[tyr-1][txr]
            + tile[tyr][txr-1]);
        current += (tile[tyr][txr+2]
            + tile[tyr+2][txr] + tile[tyr-2][txr]
            + tile[tyr][txr-2]);
        current += (tile[tyr][txr+3]
            + tile[tyr+3][txr] + tile[tyr-3][txr]
            + tile[tyr][txr-3]);
        current += (tile[tyr][txr+4]
            + tile[tyr+4][txr] + tile[tyr-4][txr]
            + tile[tyr][txr-4]);
        float temp = tile[tyr][txr];
        behind4 += temp;
        behind3 += temp;
        behind2 += temp;
        behind1 += temp;
        infront1 +=  temp;
        infront2 +=  temp;
        infront3 +=  temp;
        infront4 +=  temp;
        if(z > 2*radius){
            g_output[in_idx] = behind4;
        }
        // for(int i = 0; i < 2*radius; i++){
        //     s_out[ty][tx][i] = s_out[ty][tx][1+i];
        // }
        behind4 = behind3;
        behind3 = behind2;
        behind2 = behind1;
        behind1 = current;
        current = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = 0;  
        in_idx += stride;
        tid_in_idx += stride;
    }

    // for(int i = 1; i < radius; i++){
    //     g_output[in_idx + (dimz-radius+i-1)*stride] =  s_out[ty][tx][i-1];
    // }
    // in_idx += stride;
    g_output[in_idx] = behind4;
    in_idx += stride;
    g_output[in_idx] = behind3;
    in_idx += stride;
    g_output[in_idx] = behind2;
    in_idx += stride;
    g_output[in_idx] = behind1;
    
}








float* myStencil32(float* data, int dimx, int dimy, int dimz){
    float* out = (float*)malloc(dimx*dimy*dimz*sizeof(float));
    for(int i = 0; i < dimx*dimy*dimz; i++){
        out[i] = 0;
    }
    float *dev_data, *dev_out;
    gpuErrchk(cudaMalloc((void**)&dev_data, dimx*dimy*dimz*sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&dev_out, dimx*dimy*dimz*sizeof(float)))

    gpuErrchk(cudaMemcpy(dev_data, data, dimx*dimy*dimz*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_out, out, dimx*dimy*dimz*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocks((dimx-2*4)/32, (dimy-2*4)/32, 1);
    dim3 threads(32, 32, 1);
    my_32_4<<<blocks,threads>>>(dev_data, dev_out, dimx,dimy, dimz);
    gpuErrchk(cudaMemcpy(out, dev_out, dimx*dimy*dimz*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree((void**)dev_data));
    gpuErrchk(cudaFree((void**)dev_out));
    return out;
}