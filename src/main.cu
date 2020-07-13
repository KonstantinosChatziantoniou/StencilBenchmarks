#include <stdio.h>
#include <stdlib.h>
#include "../headers/kernel_nvidia.h"
#include "../headers/kernel_nvidia_4.h"
#include "../headers/kernel_nvidia_32.h"

#include "../headers/my_kernel_regs.h"
#include "../headers/my_kernel_regs_4.h"
#include "../headers/my_kernel_regs_32.h"


float* createData(int x,int y,int z){
    float* data = (float*)malloc(x*y*z*sizeof(int));
    for(int i = 0; i < x*y*z; i++){
        data[i] = 0;
    }
    for(int i = x/2 - 5; i < x/2+5; i++){
        for(int j = y/2 - 5; j < y/2+5; j++){
            for(int k = z/2 - 5; k < z/2+5; k++){
                //printf("%d %d %d, \n", i,j,k);
                data[i + j*x + k*x*y] = 1;
            }
        }
    }
    return data;
}

void printData(float* data, int dimx, int dimy, int dimz){
    for(int k = 0; k < dimz; k++){
        printf("-------- z = %d ------------\n",k);
        for(int i = 0; i < dimx; i++){
            for(int j = 0; j < dimy; j++){
                //printf("%d %d %d, ", i,j,k);
                printf("%f, ", data[i + j*dimx + k*dimx*dimy]);
                
            }
            printf("\n");
        }
    }
}

float* cpu_stencil(float* data, int dimx, int dimy, int dimz, int radius){
    float* c_coef = (float*)malloc((2*radius + 1)*sizeof(float));
    for(int i = 0; i <2*radius+1; i++){
        c_coef[i] = 1;
    }
    float* out = (float*)malloc(dimx*dimy*dimz*sizeof(float));
     for(int i = 0; i < dimx*dimy*dimz; i++){
        out[i] = 0;
    }
    for(int i = radius; i < dimx - radius; i++){
        for(int j = radius; j < dimy - radius; j++){
            for(int k = radius; k < dimz - radius; k++){
                int ystr = dimx;
                int zstr = dimx*dimy;
                int  index = i + j*dimx + k*dimx*dimy;
                //printf("ix %d,  i %d,  j%d,  k %d\n", k, j , index + 4*zstr ,i );
                float temp = 2.f*data[index];// - g_output[out_idx];
                float sdiv = c_coef[0] * data[index];
                for(int ix = 1; ix < radius+1; ix++){
                    sdiv += c_coef[ix]*(data[index + ix] + data[index - ix] + data[index+ix*ystr] + data[index - ix*ystr]
                            + data[index + ix*zstr] + data[index - ix*zstr]);
                }
                out[index] = temp + sdiv;
            }
        }
    }
    return out;
}

void check(float* d1, float* d2, int dimx, int dimy, int dimz){

    for(int k = 0; k < dimz; k++){
        for(int i = 0; i < dimx; i++){
            for(int j = 0; j < dimy; j++){
                if(d1[i + j*dimx + k*dimx*dimy] != d2[i + j*dimx + k*dimx*dimy]){
                    printf("ERRR %d %d %d  %f vs %f\n",i,j,k,d1[i + j*dimx + k*dimx*dimy],d2[i + j*dimx + k*dimx*dimy]);
                    return;
                }
               
                
            }
        }
    }
    printf("CORREEEEEEEEEEEEECT\n");
}
int main(int argc, char** argv){
    int radius = 4;
    int n_sz = 8;
    int x = (1<<atoi(argv[1])) + 2*radius;
    int y = (1<<atoi(argv[2])) + 2*radius;
    int z = (1<<atoi(argv[3])) + 2*radius;

    printf("Data %d %d %d\n", x, y, z);
    float* data = createData(x,y,z);
    //printf("GPU DONE\n");
    //printData(data, x, y ,z);
    // printf("GPU DONE\n");
    float* cp_out  = cpu_stencil(data, x, y ,z, 4);

    float* out1 = nvidiaStencil(data, x, y ,z);
    check(out1, cp_out, x, y, z);
    free(out1);
    float* out2 = nvidiaStencil_32_4(data, x, y ,z);
    check(out2, cp_out, x, y, z);
    free(out2);

    float* out3 = myStencil(data, x, y, z);
    check(out3, cp_out,x, y ,z);
    free(out3);
    float* out4 = myStencil32(data, x, y, z);
    check(out4, cp_out,x, y ,z);
    free(out4);
    free(cp_out);
    free(data);
    //////
    radius = 8;
    x = (1<<atoi(argv[1])) + 2*radius;
    y = (1<<atoi(argv[2])) + 2*radius;
    z = (1<<atoi(argv[3])) + 2*radius;
    float* data2 = createData(x,y,z);
    //printData(data2, x, y ,z);
    printf("Data %d %d %d\n", x, y, z);
    float* cp_out8  = cpu_stencil(data2, x, y ,z, 8);
    float* out5 = nvStencil32(data2, x, y ,z);
    check(out5, cp_out8, x, y, z);
    free(out5);
    float* out6 = myStencil32_8(data2, x, y, z);
    check(out6, cp_out8,x, y ,z);
    // printData(out4, x, y ,z);
    // printf("Data %d %d %d\n", x, y, z);
    // printData(cp_out8, x, y ,z);
    // printf("Data %d %d %d\n", x, y, z);
    return 0;
}
