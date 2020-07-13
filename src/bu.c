
__global__ void fwd_st(float* g_input, float* g_output, // float* g_vsg, = 1
                        const int dimx, const int dimy, const int dimz)
{
    #define BDIMX 16 // tile (and threadblock) size in x
    #define BDIMY 16 // tile (and threadblock) size in y
    #define radius 4 // half of the order in space (k/2)

    __shared__ float s_data[BDIMY+2*radius][BDIMX + 2*radius];

    int ix = blockIdx.x*blockDim.x + threadIdx.x  + radius;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + radius;
    int in_idx = iy*dimx + ix;                  // index for reading input
    int out_idx = 0;                            // index for writing output
    int stride = dimx*dimy;                     // distance between 2D slices (in elements)

    float infront1, infront2, infront3, infront4;
    float behind1, behind2, behind3, behind4;
    float current;

    int tx = threadIdx.x + radius;              // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    int ty = threadIdx.y + radius;

    // fill the "in-front" and "behind" data
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

    for(int i = radius; i < dimz - radius; i++){
        // advance the slice (move the thread-front)
        behind4 = behind3;
        behind3 = behind2;
        behind2 = behind1;
        behind1 = current;
        current = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = g_input[in_idx];

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
        g_output[out_idx] =  temp + sdiv;
        // if(threadIdx.x == 0 && threadIdx.y==0){
        //     printf("%d %d %f %f\n", blockIdx.x, blockIdx.y, sdiv, temp);
        // }
    }
}



__global__ void fwd_st2(float* g_input, float* g_output, // float* g_vsg, = 1
                        const int dimx, const int dimy, const int dimz)
{
    #define BDIMX 16 // tile (and threadblock) size in x
    #define BDIMY 16 // tile (and threadblock) size in y
    #define radius 4 // half of the order in space (k/2)

    __shared__ float s_data[BDIMY+2*radius][BDIMX + 2*radius];
    __shared__ float all_z_data[BDIMX*BDIMY][2*radius];
    float* z_data = all_z_data[threadIdx.x + threadIdx.y*BDIMX];
    int ix = blockIdx.x*blockDim.x + threadIdx.x  + radius;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + radius;
    int in_idx = iy*dimx + ix;                  // index for reading input
    int out_idx = 0;                            // index for writing output
    int stride = dimx*dimy;                     // distance between 2D slices (in elements)

    
    float current;

    int tx = threadIdx.x + radius;              // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    int ty = threadIdx.y + radius;

    // fill the "in-front" and "behind" data
    z_data[1] = g_input[in_idx];
    in_idx += stride;
    z_data[2] = g_input[in_idx];
    in_idx += stride;
    z_data[3] = g_input[in_idx];
    in_idx += stride;
    current = g_input[in_idx];
    out_idx = in_idx; 
    in_idx += stride;
    z_data[4] = g_input[in_idx];
    in_idx += stride;
    z_data[5] = g_input[in_idx];
    in_idx += stride;
    z_data[6] = g_input[in_idx];
    in_idx += stride;
    z_data[7] = g_input[in_idx];
    in_idx += stride;

    for(int i = radius; i < dimz - radius; i++){
        // advance the slice (move the thread-front)
        z_data[0] = z_data[1];
        z_data[1] = z_data[2];
        z_data[2] = z_data[3];
        z_data[3] = current;
        current = z_data[4];
        z_data[4] = z_data[5];
        z_data[5] = z_data[6];
        z_data[6] = z_data[7];
        z_data[7] = g_input[in_idx];

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
        // sdiv += (float)(8/5)*( z_data[4] + z_data[3]
        //     + s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] );
        // sdiv += (float)(-1/5)*( z_data[5] + z_data[2]
        //     + s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] );
        // sdiv += (float)(8/315)*( z_data[6] + z_data[1]
        //     + s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] );
        // sdiv += (float)(-1/560)*( z_data[7] + z_data[0]
        //     + s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] );

        float sdiv = current;
        sdiv += ( z_data[4] + z_data[3]
            + s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] );
        sdiv += ( z_data[5] + z_data[2]
            + s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] );
        sdiv += ( z_data[6] + z_data[1]
            + s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] );
        sdiv += ( z_data[7] + z_data[0]
            + s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] );
        g_output[out_idx] =  temp + sdiv;
        // if(threadIdx.x == 0 && threadIdx.y==0){
        //     printf("%d %d %f %f\n", blockIdx.x, blockIdx.y, sdiv, temp);
        // }
    }
}



__global__ void fwd_st3(float* g_input, float* g_output, // float* g_vsg, = 1
                        const int dimx, const int dimy, const int dimz)
{
    #define BDIMX 16 // tile (and threadblock) size in x
    #define BDIMY 16 // tile (and threadblock) size in y
    #define radius 4 // half of the order in space (k/2)

    __shared__ float tile[BDIMY+2*radius][BDIMX + 2*radius];
    __shared__ float s_out[BDIMX*BDIMY][2*radius + 1];
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
    // Init out shared array
    for(int i = 0; i < (2*radius+1); i++){
        s_out[tx + ty*BDIMX][i] = 0;   
    }

    for(int z = radius; z < dimz - radius; z++){
        // read tile with halo
        for(int i = 0; i < len; i += BDIMX){
            for(int j = 0; j < len; j += BDIMX){
                if(tx + i < len && ty + j < len)
                    tile[tx + i][ty+j] = g_input[in_idx + z*stride + i + j*dimx - radius*(dimx+1)];
            }
        }

        __syncthreads();
        s_out[tx + ty*BDIMX][4] += 3*tile[tx+radius][ty+radius];
        for(int i = 1; i <= radius; i++){
            s_out[tx+ty*BDIMX][4] += (tile[tx+radius+i][ty+radius]
            + tile[tx+radius-i][ty+radius] + tile[tx+radius][ty+radius+i]
            + tile[tx+radius][ty+radius-i]);

            s_out[tx+ty*BDIMX][4+i] += tile[tx+radius][ty+radius];
            s_out[tx+ty*BDIMX][i-1] += tile[tx+radius][ty+radius];
        }

        if(z > 2*radius){
            g_output[in_idx + (z-radius)*stride] = s_out[tx + ty*BDIMX][0];
        }
        for(int i = 0; i < 2*radius; i++){
            s_out[tx + ty*BDIMX][i] = s_out[tx + ty*BDIMX][1+i];
        }
        s_out[tx + ty*BDIMX][8] = 0;
        __syncthreads();
    }

    for(int i = 1; i < radius; i++){
        g_output[in_idx + (dimz-radius+i-1)*stride] =  s_out[tx + ty*BDIMX][i-1];
    }
    
}




