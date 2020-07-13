#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include "../headers/kernel_nvidia_32.h"

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




__global__ void nv_32_8(float* g_input, float* g_output, // float* g_vsg, = 1
                        const int dimx, const int dimy, const int dimz)
{
    #define BDIMX 32 // tile (and threadblock) size in x
    #define BDIMY 32 // tile (and threadblock) size in y
    #define radius 8 // half of the order in space (k/2)

    __shared__ float s_data[BDIMY+2*radius][BDIMX + 2*radius];

    int ix = blockIdx.x*blockDim.x + threadIdx.x  + radius;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + radius;
    int in_idx = iy*dimx + ix;                  // index for reading input
    int out_idx = 0;                            // index for writing output
    int stride = dimx*dimy;                     // distance between 2D slices (in elements)

    float infront1, infront2, infront3, infront4, infront5, infront6, infront7, infront8;
    float behind1, behind2, behind3, behind4, behind5, behind6, behind7, behind8;
    float current;

    int tx = threadIdx.x + radius;              // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    int ty = threadIdx.y + radius;

    // fill the "in-front" and "behind" data
   
    behind7 = g_input[in_idx];
    in_idx += stride;
    behind6 = g_input[in_idx];
    in_idx += stride;
    behind5 = g_input[in_idx];
    in_idx += stride;
    behind4 = g_input[in_idx];
    in_idx += stride;
    behind3 = g_input[in_idx];
    in_idx += stride;
    behind2 = g_input[in_idx];
    in_idx += stride;
    behind1 = g_input[in_idx];
    in_idx += stride;
    current = g_input[in_idx];
    out_idx = in_idx; 
    in_idx += stride;
    infront1 = g_input[in_idx];
    in_idx += stride;
    infront2 = g_input[in_idx];
    in_idx += stride;
    infront3 = g_input[in_idx];
    in_idx += stride;
    infront4 = g_input[in_idx];
    in_idx += stride;
    infront5 = g_input[in_idx];
    in_idx += stride;
    infront6 = g_input[in_idx];
    in_idx += stride;
    infront7 = g_input[in_idx];
    in_idx += stride;
    infront8 = g_input[in_idx];
    in_idx += stride;

    for(int i = radius; i < dimz - radius; i++){
        // advance the slice (move the thread-front)
        behind8 = behind7;
        behind7 = behind6;
        behind6 = behind5;
        behind5 = behind4;
        behind4 = behind3;
        behind3 = behind2;
        behind2 = behind1;
        behind1 = current;
        current = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = infront5;
        infront5 = infront6;
        infront6 = infront7;
        infront7 = infront8;
        infront8 = g_input[in_idx];

        in_idx += stride;
        out_idx += stride;
        __syncthreads();
        /////////////////////////////////////////
        // update the data slice in smem
        if(threadIdx.y<radius)
        // halo above/below
        {
            s_data[threadIdx.y][tx] = g_input[out_idx-radius*dimx];
            s_data[threadIdx.y+BDIMY+radius][tx] = g_input[out_idx+BDIMY*dimx];
        }
        if(threadIdx.x<radius)
        // halo left/right
        {
            s_data[ty][threadIdx.x] = g_input[out_idx-radius];
            s_data[ty][threadIdx.x+BDIMX+radius] = g_input[out_idx+BDIMX];
        }
        // update the slice in smem
        s_data[ty][tx] = current;
        __syncthreads();
        /////////////////////////////////////////
        // compute the output value
        float temp = 2.f*current;// - g_output[out_idx];
        // float sdiv = (float)(-205/72) * current;
        // sdiv += (float)(8/5)*( infront1 + behind1
        //     + s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] );
        // sdiv += (float)(-1/5)*( infront2 + behind2
        //     + s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] );
        // sdiv += (float)(8/315)*( infront3 + behind3
        //     + s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] );
        // sdiv += (float)(-1/560)*( infront4 + behind4
        //     + s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] );

        float sdiv = current;
        sdiv += ( infront1 + behind1
            + s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] );
        sdiv += ( infront2 + behind2
            + s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] );
        sdiv += ( infront3 + behind3
            + s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] );
        sdiv += ( infront4 + behind4
            + s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] );
        sdiv += ( infront5 + behind5
            + s_data[ty-5][tx] + s_data[ty+5][tx] + s_data[ty][tx-5] + s_data[ty][tx+5] );
        sdiv += ( infront6 + behind6
            + s_data[ty-6][tx] + s_data[ty+6][tx] + s_data[ty][tx-6] + s_data[ty][tx+6] );
        sdiv += ( infront7 + behind7
            + s_data[ty-7][tx] + s_data[ty+7][tx] + s_data[ty][tx-7] + s_data[ty][tx+7] );
        sdiv += ( infront8 + behind8
            + s_data[ty-8][tx] + s_data[ty+8][tx] + s_data[ty][tx-8] + s_data[ty][tx+8] );
        g_output[out_idx] =  temp + sdiv;
        // if(threadIdx.x == 0 && threadIdx.y==0){
        //     printf("%d %d %f %f\n", blockIdx.x, blockIdx.y, sdiv, temp);
        // }
    }
}








float* nvStencil32(float* data, int dimx, int dimy, int dimz){
    float* out = (float*)malloc(dimx*dimy*dimz*sizeof(float));
    for(int i = 0; i < dimx*dimy*dimz; i++){
        out[i] = 0;
    }
    float *dev_data, *dev_out;
    gpuErrchk(cudaMalloc((void**)&dev_data, dimx*dimy*dimz*sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&dev_out, dimx*dimy*dimz*sizeof(float)))

    gpuErrchk(cudaMemcpy(dev_data, data, dimx*dimy*dimz*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_out, out, dimx*dimy*dimz*sizeof(float), cudaMemcpyHostToDevice));

    //int nblocks = (dimx-2*4)/16;
    dim3 blocks((dimx-2*8)/32, (dimy-2*8)/32, 1);
    dim3 threads(32, 32, 1);
    nv_32_8<<<blocks,threads>>>(dev_data, dev_out, dimx,dimy, dimz);
    gpuErrchk(cudaMemcpy(out, dev_out, dimx*dimy*dimz*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree((void**)dev_data));
    gpuErrchk(cudaFree((void**)dev_out));
    return out;
}