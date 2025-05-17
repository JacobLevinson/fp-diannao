#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

#define WEIGHT_BYTES   (SYNAPSE_SIZE * sizeof(VTYPE))
#define INPUT_BYTES    (NYPAD * NXPAD * Ni * sizeof(VTYPE))
#define OUTPUT_BYTES   (NYSCL * NXSCL * Nn * sizeof(VTYPE))
#define OUTPUT_ELEMENTS (NYSCL * NXSCL * Nn)

#define BX 8
#define BY 8
#define BZ 8          // your 8×8×8 launch

// Note: VTYPE is a float
VTYPE (*synapse)[Ky][Kx][Nn][Ni];

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

// Array initialization
void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

// Convolution with Tiling
void convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

// Simple Convolution
void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                        VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                        VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}
// ---- forward declaration ------------------------------------
__global__ void convolution_layer_kernel(VTYPE*, VTYPE*, VTYPE*);

float run_kernel(int iters,
                 VTYPE* d_syn, VTYPE* d_in, VTYPE* d_out,
                 dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float total_ms = 0.0f;

    for (int it = 0; it < iters; ++it)
    {
        cudaMemcpyAsync(d_in, neuron_i, 
                        INPUT_BYTES,  // = NYPAD*NXPAD*Ni*sizeof(VTYPE)  or  Ni*sizeof(VTYPE)
                        cudaMemcpyHostToDevice);

        cudaEventRecord(t0);
        convolution_layer_kernel<<<grid, block>>>(d_syn, d_in, d_out);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;  cudaEventElapsedTime(&ms, t0, t1);
        total_ms += ms;
    }
    return total_ms / iters;
}

// GPU Code
__global__ void convolution_layer_kernel(VTYPE* synapse,
                                         VTYPE* neuron_i,
                                         VTYPE* neuron_n)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_n = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx_x >= Nx || idx_y >= Ny || idx_n >= Nn) return;

    int xout = idx_x / Sx;
    int yout = idx_y / Sy;

    VTYPE sum = 0.0f;

    // Same two kernel loops as before, but use idx_n instead of looping over n
    for (int ky = 0; ky < Ky; ky++)
        for (int kx = 0; kx < Kx; kx++)
            for (int i = 0; i < Ni; i++) {
                int synapse_idx =
                     ((ky * Kx + kx) * Nn + idx_n) * Ni + i;
                int neuron_i_idx =
                     ((ky + idx_y) * NXPAD + kx + idx_x) * Ni + i;
                VTYPE sv = synapse[synapse_idx];
                VTYPE nv = neuron_i[neuron_i_idx];
                sum += sv * nv;
            }

    int neuron_n_idx = ((yout * NXSCL + xout) * Nn + idx_n);
    neuron_n[neuron_n_idx] = (sum > 0) ? sum : sum / 4.0f;
}

// CUDA Kernel with Host Code
// void convolution_layer_cuda(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
//                             VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
//                             VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {

//     // Allocate space for device copies of synapse, neuron_i, and neuron_n
//     VTYPE *d_synapse, *d_neuron_i, *d_neuron_n;
//     cudaMalloc((void**)&d_synapse, SYNAPSE_SIZE*sizeof(VTYPE));
//     cudaMalloc((void**)&d_neuron_i, NYPAD*NXPAD*Ni*sizeof(VTYPE));
//     cudaMalloc((void**)&d_neuron_n, NYSCL*NXSCL*Nn*sizeof(VTYPE));
    
//     // Copy inputs to device
//     cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_neuron_i, neuron_i, NYPAD*NXPAD*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);
    
//     // // Launch the kernel
//     // dim3 blockDim(2, 2); // Hardcoded for Nx and Ny
//     // dim3 gridDim(Nx / blockDim.x, Ny / blockDim.y); // 7x7
//     // convolution_layer_kernel<<<gridDim, blockDim>>>(d_synapse, d_neuron_i, d_neuron_n);


//     dim3 blockDim(BX, BY, BZ);                 // 8 × 8 × 8 = 512 threads / block

//     // gridDim.x  covers the 14×14 spatial map
//     // gridDim.z  walks through the 512 output channels in chunks of 8
//     dim3 gridDim( (Nx + BX - 1) / BX,          // ceil(Nx / BX)  →  2
//                   (Ny + BY - 1) / BY,          // ceil(Ny / BY)  →  2
//                   (Nn + BZ - 1) / BZ );        // ceil(512 / 8)  →  64

//     convolution_layer_kernel<<< gridDim, blockDim >>>(d_synapse,
//                                                       d_neuron_i,
//                                                       d_neuron_n);
//     cudaDeviceSynchronize();


//     // Copy result back to host
//     cudaMemcpy(neuron_n, d_neuron_n, NYSCL*NXSCL*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

//     // Cleanup
//     cudaFree(d_synapse);
//     cudaFree(d_neuron_i);
//     cudaFree(d_neuron_n);
// }

// Main Execution
int main(const int argc, const char** argv) {
  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  // ---------------- one-time setup ----------------
  fill_convolution_shared_simple(*synapse, *neuron_i);

  VTYPE *d_syn, *d_in, *d_out;
  cudaMalloc(&d_syn, WEIGHT_BYTES);          // e.g. SYNAPSE_SIZE*sizeof(VTYPE)
  cudaMalloc(&d_in , INPUT_BYTES);           // size of one activation tensor
  cudaMalloc(&d_out, OUTPUT_BYTES);          // NYSCL*NXSCL*Nn*sizeof(VTYPE)

  cudaMemcpy(d_syn, synapse, WEIGHT_BYTES, cudaMemcpyHostToDevice);

  // pick grid/block that you already computed
  dim3 block(BX, BY, BZ);
  dim3 grid((Nx+BX-1)/BX, (Ny+BY-1)/BY, (Nn+BZ-1)/BZ);

  // ---------------- timed region ------------------
  int ITERS = 30;
  float ms = run_kernel(ITERS, d_syn, d_in, d_out, grid, block);
  printf("KERNEL time (avg of %d): %.3f ms\n", ITERS, ms);

  // optional correctness
  convolution_layer_blocked(*synapse, *neuron_i, *neuron_n2);
  cudaMemcpy(neuron_n, d_out, OUTPUT_BYTES, cudaMemcpyDeviceToHost);
  compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, OUTPUT_ELEMENTS);

  // cleanup
  cudaFree(d_syn); cudaFree(d_in); cudaFree(d_out);
}


